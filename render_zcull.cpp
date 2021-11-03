#include <array>
#include <vector>
#include <atomic>
#include <limits>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <memory>
#include <cstdint>
#include <omp.h>

typedef float coord_t;

struct Vertex {
    coord_t x, y, z;
    
    template <typename T>
    Vertex operator*(const T &i) const {
        return Vertex {this->x*i, this->y*i, this->z*i};
    }
    template <typename T>
    Vertex operator*(std::array<T,3> i) const {
        return Vertex {this->x*i[0], this->y*i[1], this->z*i[2]};
    }
};

struct Triangle {
    Vertex v0, v1, v2;
    
    template <typename T>
    Triangle operator*(const T &i) const {
        return Triangle {this->v0*i, this->v1*i, this->v2*i};
    }
    template <typename T>
    Triangle operator*(std::array<T,3> i) const {
        return Triangle {this->v0*i[0], this->v1*i[1], this->v2*i[2]};
    }
};

struct Pixel {
    uint8_t r, g, b;
};

typedef std::vector<std::vector<std::shared_ptr<std::atomic<coord_t> > > > buffer_t;
typedef std::vector<std::vector<Pixel > > bitmap_data_t;

coord_t edgeFunction(const Vertex a, const Vertex b, const Vertex c) {
    return ((c.x-a.x)*(b.y-a.y) - (c.y-a.y)*(b.x-a.x));
}

// Adaptation of the rasterization technique discussed in "A Parallel Algorithm for Polygon Rasterization", Juan Pineda, 1988
// Note that despite the name, this is a sequential implementation - the very fine-grain parallelism is likely to be dominated by overhead on a CPU
void rasterize_triangle(const Triangle &tri, buffer_t &zbuf) {
    const size_t y_res = zbuf.size();
    const size_t x_res = zbuf[0].size();
    
    //Precompute some constants which will be used for depth calculations
    coord_t z0 = tri.v0.z;
    coord_t z0_z1_delta = tri.v1.z - z0;
    coord_t z0_z2_delta = tri.v2.z - z0;

    // Scale the triangle into the raster space
    std::array<size_t, 3> raster_scalar = {x_res, y_res, 1};
    Triangle raster_tri = tri * raster_scalar;
    
    // Get the area of the parallelogram defined by edges (v0, v1) and (v0, v2)
    //  This is twice the area of the raster triangle
    coord_t twice_area = edgeFunction(raster_tri.v0, raster_tri.v1, raster_tri.v2);

    // Get a 2D bounding box for the triangle; this is a quick way to reduce the search space
    // TODO: There are better techniques than a bounding box; see the center line algorithm in the paper (Fig. 5)
    int32_t x_min = std::min({raster_tri.v0.x, raster_tri.v1.x, raster_tri.v2.x});
    x_min = std::clamp(x_min, 0, (int32_t)(x_res-1));
    int32_t x_max = std::max({raster_tri.v0.x, raster_tri.v1.x, raster_tri.v2.x});
    x_max = std::clamp(x_max, 0, (int32_t)(x_res-1));
    int32_t y_min = std::min({raster_tri.v0.y, raster_tri.v1.y, raster_tri.v2.y});
    y_min = std::clamp(y_min, 0, (int32_t)(y_res-1));
    int32_t y_max = std::max({raster_tri.v0.y, raster_tri.v1.y, raster_tri.v2.y});
    y_max = std::clamp(y_max, 0, (int32_t)(y_res-1));

    // Test membership in the triangle for each pixel in the bounding box
    for (int32_t x = x_min; x <= x_max; x++) {
        for (int32_t y = y_min; y <= y_max; y++) {
            Vertex point {(coord_t)x, (coord_t)y, 0};
            // These three edge functions partition the plane along three lines
            // If a point is to the left of a line, the corresponding edge function will evaluate negative
            // If the point is on or to the right of the line, the edge function will be >= 0
            // Since vertices are defined clockwise, a point is inside the triangle iff all functions evaluate positive
            coord_t w0 = edgeFunction(raster_tri.v1, raster_tri.v2, point);
            coord_t w1 = edgeFunction(raster_tri.v2, raster_tri.v0, point);
            coord_t w2 = edgeFunction(raster_tri.v0, raster_tri.v1, point);
            if ((w0 >= 0) && (w1 >= 0) && (w2 >= 0)) { // Point is inside the triangle
                // w0, w1, and w2 are denormalized, but their normalized values sum to 1
                // We can exploit this to simplify the depth calculation (w0*z0 + w1*z1 + w2*z2) by eliminating w0
                coord_t norm_w1 = w1 / twice_area;
                coord_t norm_w2 = w2 / twice_area;
                coord_t depth = z0 + norm_w1*z0_z1_delta + norm_w2*z0_z2_delta;
                // Atomic compare and swap if less than
                bool done = false;
                do {
                    coord_t current = zbuf[y][x]->load();
                    if (depth < current) {
                        done = zbuf[y][x]->compare_exchange_weak(current, depth);
                    } else {
                        done = true;
                    }
                } while (!done);
            }
        }
    }
}

buffer_t rasterize_scene(std::vector<Triangle> tris, const size_t x_res, const size_t y_res, const size_t num_threads) {
    // Construct the z-buffer
    buffer_t zbuf;
    for (size_t y = 0; y < y_res; y++) {
        std::vector<std::shared_ptr<std::atomic<coord_t> > > row;
        for (size_t x = 0; x < x_res; x++) {
            std::atomic<coord_t> *obj = new std::atomic<coord_t> (std::numeric_limits<coord_t>::infinity());
            row.push_back(std::shared_ptr<std::atomic<coord_t> > (obj));
        }
        zbuf.push_back(row);
    }
    
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (auto &tri : tris) {
        rasterize_triangle(tri, zbuf);
    }

    return zbuf;
}

bitmap_data_t buffer_to_image_data(const buffer_t zbuf) {
    const size_t y_res = zbuf.size();
    const size_t x_res = zbuf[0].size();
    const coord_t infty = std::numeric_limits<coord_t>::infinity();

    coord_t min_depth = infty;
    coord_t max_depth = -infty;
    for (size_t y = 0; y < y_res; y++) {
        for (size_t x = 0; x < x_res; x++) {
            coord_t depth = zbuf[y][x]->load();
            if ((depth > -100) && (depth < infty)) {
                if (depth < min_depth) min_depth = depth;
                if (depth > max_depth) max_depth = depth;
            }
        }
    }

    bitmap_data_t image;
    for (size_t y = 0; y < y_res; y++) {
        std::vector<Pixel> row;
        for (size_t x = 0; x < x_res; x++) {
            Pixel pixel;
            coord_t depth = zbuf[y][x]->load();
            if (depth <= -100) {
                pixel.r = 0x7f;
                pixel.g = 0x7f;
                pixel.b = 0x7f;
            } else if (depth == infty) {
                pixel.r = 0xff;
                pixel.g = 0xff;
                pixel.b = 0xff;
            } else {
                float scaled = (depth-min_depth)/(max_depth-min_depth);
                pixel.r = 0xff*(1-scaled);
                pixel.g = 0x00;
                pixel.b = 0xff*scaled;
            }
            row.push_back(pixel);
        }
        image.push_back(row);
    }

    return image;
}

// https://stackoverflow.com/questions/2654480/writing-bmp-image-in-pure-c-c-without-other-libraries
void save_bitmap(bitmap_data_t image, const char* fname) {
    size_t height = image.size();
    size_t width = image[0].size();

    FILE *f;
    size_t data_size = 3*width*height;
    size_t filesize = data_size + 54; // Includes header size
    uint8_t *img = (uint8_t*)malloc(data_size);

    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            size_t x = j;
            size_t y = (height-1)-i; // Since bitmaps are specified top-to-bottom
            img[(x+y*width)*3+2] = image[i][j].r;
            img[(x+y*width)*3+1] = image[i][j].g;
            img[(x+y*width)*3+0] = image[i][j].b;
        }
    }

    // Construct header
    uint8_t bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    uint8_t bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    uint8_t bmppad[3] = {0,0,0};

    bmpfileheader[ 2] = (uint8_t)(filesize    );
    bmpfileheader[ 3] = (uint8_t)(filesize>> 8);
    bmpfileheader[ 4] = (uint8_t)(filesize>>16);
    bmpfileheader[ 5] = (uint8_t)(filesize>>24);

    bmpinfoheader[ 4] = (uint8_t)(   width    );
    bmpinfoheader[ 5] = (uint8_t)(   width>> 8);
    bmpinfoheader[ 6] = (uint8_t)(   width>>16);
    bmpinfoheader[ 7] = (uint8_t)(   width>>24);
    bmpinfoheader[ 8] = (uint8_t)(  height    );
    bmpinfoheader[ 9] = (uint8_t)(  height>> 8);
    bmpinfoheader[10] = (uint8_t)(  height>>16);
    bmpinfoheader[11] = (uint8_t)(  height>>24);

    f = fopen(fname,"wb");
    fwrite(bmpfileheader,1,14,f);
    fwrite(bmpinfoheader,1,40,f);
    for(size_t i=0; i<height; i++) {
        fwrite(img+(width*(height-i-1)*3),3,width,f);
        fwrite(bmppad,1,(4-(width*3)%4)%4,f);
    }

    free(img);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <Input .tri> <Output .bmp> <X resolution> <Y resolution> <# threads>" << std::endl;
        return 1;
    }

    // 1. Handle command-line params
    char *in_fname = argv[1];
    char *out_fname = argv[2];
    int x_res = atoi(argv[3]);
    int y_res = atoi(argv[4]);
    int num_threads = atoi(argv[5]);

    // 2. Parse triangle file
    std::ifstream infile(in_fname);
    std::string line;
    std::vector<Triangle> tris;
    while (std::getline(infile, line)) {
        std::array<coord_t,9> tbuff;
        size_t idx = 0;
        std::stringstream ss(line);
        for (coord_t c; ss >> c;) {
            if (idx > 8) {
                std::cerr << "ERROR: Malformatted input line: Too many values" << std::endl;
                return 2;
            }
            tbuff[idx++] = c;
            if (ss.peek() == ',')
                ss.ignore();
        }
        if (idx < 9) {
            std::cerr << "ERROR: Malformatted input line: Too few values" << std::endl;
            return 2;
        }
        Vertex v0 {tbuff[0], tbuff[1], tbuff[2]};
        Vertex v1 {tbuff[3], tbuff[4], tbuff[5]};
        Vertex v2 {tbuff[6], tbuff[7], tbuff[8]};
        Triangle tri {v0, v1, v2};
        tris.push_back(tri);
    }

    // 3. Rasterize triangles
    buffer_t zbuf = rasterize_scene(tris, x_res, y_res, num_threads);
    /*
    for (size_t y = 0; y < y_res; y++) {
        for (size_t x = 0; x < x_res; x++) {
            std::cout << zbuf[y][x]->load() << " ";
        }
        std::cout << std::endl;
    }
    */

    // 4. Shade pixels
    bitmap_data_t image = buffer_to_image_data(zbuf);

    // 5. Save bitmap
    save_bitmap(image, out_fname);

    return 0;
}

