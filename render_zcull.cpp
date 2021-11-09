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
#include <ctime>
#include <omp.h>
#include <math.h>

typedef float coord_t;

enum raster_mode_t {
    NAIVE,
    BOUNDING_BOX,
    INTERPOLATED
};

struct Vertex {
    coord_t x, y, z;
    
    template <typename T>
    Vertex operator*(const T &i) const {
        return Vertex {this->x*i, this->y*i, this->z*i};
    }
    template <typename T>
    Vertex operator*(std::array<T,3> &i) const {
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
    Triangle operator*(std::array<T,3> &i) const {
        return Triangle {this->v0*i, this->v1*i, this->v2*i};
    }
};

struct Pixel {
    uint8_t r, g, b;
};

typedef std::vector<std::vector<std::shared_ptr<std::atomic<coord_t> > > > buffer_t;
typedef std::vector<std::vector<Pixel > > bitmap_data_t;
typedef std::vector<std::vector<coord_t>> s_buffer;

coord_t edgeFunction(const Vertex a, const Vertex b, const Vertex c) {
    return ((c.x-a.x)*(b.y-a.y) - (c.y-a.y)*(b.x-a.x));
}

bool rasterize_pixel(
    const Triangle &raster_tri, buffer_t &zbuf,
    coord_t twice_area, coord_t z0, coord_t z0_z1_delta, coord_t z0_z2_delta,
    int32_t x, int32_t y, s_buffer &OM, bool update_hom
) {
    const size_t y_res = zbuf.size();
    const size_t x_res = zbuf[0].size();
    Vertex point {(coord_t)x, (coord_t)y, 0};
    // These three edge functions partition the plane along three lines
    // If a point is to the left of a line, the corresponding edge function will evaluate negative
    // If the point is on or to the right of the line, the edge function will be >= 0
    // Since vertices are defined clockwise, a point is inside the triangle iff all functions evaluate positive
    coord_t w0 = edgeFunction(raster_tri.v1, raster_tri.v2, point);
    coord_t w1 = edgeFunction(raster_tri.v2, raster_tri.v0, point);
    coord_t w2 = edgeFunction(raster_tri.v0, raster_tri.v1, point);
    bool in_bounds = (w0 >= 0) && (w1 >= 0) && (w2 >= 0);
    if (in_bounds) { // Point is inside the triangle
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
                if (update_hom)
                    OM[floor(((y*1.0)/y_res) * 256)][floor((x*1.0/x_res) * 256)] = 1;
            } else {
                done = true;
            }
        } while (!done);
    }
    return in_bounds;
}

// Adaptation of the rasterization technique discussed in "A Parallel Algorithm for Polygon Rasterization", Juan Pineda, 1988
// Note that despite the name, this is a sequential implementation - the very fine-grain parallelism is likely to be dominated by overhead on a CPU
void rasterize_triangle(const Triangle &tri, buffer_t &zbuf, raster_mode_t mode, s_buffer &OM, bool update_hom) {
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

    switch(mode) {
    case NAIVE: {
        // The simplest form of rasterization, where we check every pixel for membership
        for (uint32_t x = 0; x < x_res; x++) {
            for (uint32_t y = 0; y < y_res; y++) {
                rasterize_pixel(raster_tri, zbuf, twice_area, z0, z0_z1_delta, z0_z2_delta, x, y, OM, update_hom);
            }
        }
        break;
    }
    case BOUNDING_BOX: {
        // Get a 2D bounding box for the triangle; this is a quick way to reduce the search space
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
                rasterize_pixel(raster_tri, zbuf, twice_area, z0, z0_z1_delta, z0_z2_delta, x, y, OM, update_hom);
            }
        }
        break;
    }
    case INTERPOLATED: {
        // Branch out from a central vertical line
        
        // Get a 2D bounding box to limit the search space
        int32_t x_min = std::min({raster_tri.v0.x, raster_tri.v1.x, raster_tri.v2.x});
        x_min = std::clamp(x_min, 0, (int32_t)(x_res-1));
        int32_t x_max = std::max({raster_tri.v0.x, raster_tri.v1.x, raster_tri.v2.x});
        x_max = std::clamp(x_max, 0, (int32_t)(x_res-1));
        int32_t y_min = std::min({raster_tri.v0.y, raster_tri.v1.y, raster_tri.v2.y});
        y_min = std::clamp(y_min, 0, (int32_t)(y_res-1));
        int32_t y_max = std::max({raster_tri.v0.y, raster_tri.v1.y, raster_tri.v2.y});
        y_max = std::clamp(y_max, 0, (int32_t)(y_res-1));

        bool found_first = false;
        bool swept = false;
        bool sweep_right;
        int32_t anchor_x, anchor_y;
        // Find the leftmost pixel on the first row with pixels
        for (anchor_y = y_min; anchor_y <= y_max; anchor_y++) {
            for (anchor_x = x_min; anchor_x <= x_max; anchor_x++) {
                if (rasterize_pixel(raster_tri, zbuf, twice_area, z0, z0_z1_delta, z0_z2_delta, anchor_x, anchor_y, OM, update_hom)) {
                    found_first = true;
                }
                if (found_first) break;
            }
            if (found_first) break;
        }
        if (!found_first) break; // Volume is empty??

        for (; anchor_y <= y_max; anchor_y++) {
            bool row_empty = false;
            if (!rasterize_pixel(raster_tri, zbuf, twice_area, z0, z0_z1_delta, z0_z2_delta, anchor_x, anchor_y, OM, update_hom)) {
                // Find the new anchor point
                if (!swept) { // This is the first time we've gone out of bounds
                    for (int32_t offset = 1;; offset++) {
                        if ((anchor_x-offset < x_min) && (anchor_x+offset > x_max)) {
                            row_empty = true;
                            break;
                        }
                        if (anchor_x-offset >= x_min) {
                            if (rasterize_pixel(raster_tri, zbuf, twice_area, z0, z0_z1_delta, z0_z2_delta, anchor_x-offset, anchor_y, OM, update_hom)) { // Test left
                                anchor_x -= offset;
                                swept = true;
                                sweep_right = false;
                                break;
                            }
                        }
                        if (anchor_x+offset <= x_max) {
                            if (rasterize_pixel(raster_tri, zbuf, twice_area, z0, z0_z1_delta, z0_z2_delta, anchor_x+offset, anchor_y, OM, update_hom)) { // Test right
                                anchor_x += offset;
                                swept = true;
                                sweep_right = true;
                                break;
                            }
                        }
                    }
                } else { // We've gone out of bounds before, and can just sweep in the same direction
                    if (sweep_right) {
                        while (anchor_x <= x_max) {
                            if (anchor_x == x_max) {
                                row_empty = true;
                                anchor_x = x_min;
                                break;
                            }
                            anchor_x++;
                            if (rasterize_pixel(raster_tri, zbuf, twice_area, z0, z0_z1_delta, z0_z2_delta, anchor_x, anchor_y, OM, update_hom)) {
                                break;
                            }
                        }
                    } else {
                        while (anchor_x >= x_min) {
                            if (anchor_x == x_min) {
                                row_empty = true;
                                anchor_x = x_max;
                                break;
                            }
                            anchor_x--;
                            if (rasterize_pixel(raster_tri, zbuf, twice_area, z0, z0_z1_delta, z0_z2_delta, anchor_x, anchor_y, OM, update_hom)) {
                                break;
                            }
                        }
                    }
                }
            }
            if (row_empty) continue;
            for (int32_t x = anchor_x-1; x >= x_min; x--) { // Sweep left from anchor point
                if (!rasterize_pixel(raster_tri, zbuf, twice_area, z0, z0_z1_delta, z0_z2_delta, x, anchor_y, OM, update_hom)) {
                    break;
                }
            }
            for (int32_t x = anchor_x+1; x <= x_max; x++) { // Sweep right from anchor point
                if (!rasterize_pixel(raster_tri, zbuf, twice_area, z0, z0_z1_delta, z0_z2_delta, x, anchor_y, OM, update_hom)) {
                    break;
                }
            }
        }

        break;
    }
    default:
        std::cerr << "Unrecognized raster mode" << std::endl;
        exit(1);
    }
}

coord_t avg_pixel_val(coord_t v0, coord_t v1, coord_t v2, coord_t v3) {
    float alpha = 0.5;
    float beta = 0.5;

    return (1-alpha)*(1-beta)*1.0*v0 + alpha*(1-beta)*1.0*v1 + alpha*beta*1.0*v2 + (1-alpha)*beta*1.0*v3;
}

void generate_hom_levels(std::vector<s_buffer> &hom) {
    for (long unsigned int i = 1; i < hom.size(); i++) {
        s_buffer p_map = hom[i-1];
        s_buffer &curr_map = hom[i];
        for (long unsigned int j = 0; j < curr_map.size(); j++) {
            for (long unsigned int k = 0; k < curr_map[0].size(); k++) {
                coord_t avg_val = avg_pixel_val(
                    p_map[2*j][2*k],
                    p_map[2*j+1][2*k],
                    p_map[2*j][2*k+1],
                    p_map[2*j+1][2*k+1]);
                curr_map[j][k] = avg_val;
            }
        }
    }
}

bool overlap_test(const Triangle &tri, std::vector<s_buffer> &hom, int level) {

    if (level < 0) return false;

    const size_t y_res = hom[level].size();
    const size_t x_res = hom[level][0].size();

    std::array<size_t, 3> raster_scalar = {x_res, y_res, 1};
    Triangle raster_tri = tri * raster_scalar;

    // Create a bounding box and test for all the pixels in this box - conservative overlap test
    int32_t x_min = std::min({raster_tri.v0.x, raster_tri.v1.x, raster_tri.v2.x});
    x_min = std::clamp(x_min, 0, (int32_t)(x_res-1));
    int32_t x_max = std::max({raster_tri.v0.x, raster_tri.v1.x, raster_tri.v2.x});
    x_max = std::clamp(x_max, 0, (int32_t)(x_res-1));
    int32_t y_min = std::min({raster_tri.v0.y, raster_tri.v1.y, raster_tri.v2.y});
    y_min = std::clamp(y_min, 0, (int32_t)(y_res-1));
    int32_t y_max = std::max({raster_tri.v0.y, raster_tri.v1.y, raster_tri.v2.y});
    y_max = std::clamp(y_max, 0, (int32_t)(y_res-1));

    for (int32_t x = x_min; x <= x_max; x++) {
        for (int32_t y = y_min; y <= y_max; y++) {
            if (hom[level][y][x] < 1) {
                return overlap_test(tri, hom, level - 1);
            }
        }
    }
    return true;
}

bool depth_test(const Triangle &tri, buffer_t &debuf) {

    const size_t y_res = 64;
    const size_t x_res = 64;

    std::array<size_t, 3> raster_scalar = {x_res, y_res, 1};
    Triangle raster_tri = tri * raster_scalar;

    // creates a bounding box and check if it lies behind any triangles rasterized as of now - conservative depth test
    int32_t x_min = std::min({raster_tri.v0.x, raster_tri.v1.x, raster_tri.v2.x});
    x_min = std::clamp(x_min, 0, (int32_t)(x_res-1));
    int32_t x_max = std::max({raster_tri.v0.x, raster_tri.v1.x, raster_tri.v2.x});
    x_max = std::clamp(x_max, 0, (int32_t)(x_res-1));
    int32_t y_min = std::min({raster_tri.v0.y, raster_tri.v1.y, raster_tri.v2.y});
    y_min = std::clamp(y_min, 0, (int32_t)(y_res-1));
    int32_t y_max = std::max({raster_tri.v0.y, raster_tri.v1.y, raster_tri.v2.y});
    y_max = std::clamp(y_max, 0, (int32_t)(y_res-1));

    coord_t tri_depth = std::min({tri.v0.z, tri.v1.z, tri.v2.z});
    for (int32_t x = x_min; x <= x_max; x++) {
        for (int32_t y = y_min; y <= y_max; y++) {
            coord_t z_val = debuf[y][x]->load();
            if (z_val > tri_depth) {
                return false;
            }
        }
    }

    return true;
}

void update_dbuf(const Triangle &tri, buffer_t dbuf, s_buffer &OM) {
    rasterize_triangle(tri, dbuf, INTERPOLATED ,OM, false);
}

buffer_t rasterize_scene(std::vector<Triangle> tris, const size_t x_res, const size_t y_res, const size_t num_threads, raster_mode_t mode, bool hom_enabled) {
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

    // Construct HOM - 256x256
    std::vector<s_buffer> hom;

    for (size_t i = 256; i > 1; i /= 2){
        s_buffer OM;
        for (size_t y = 0; y < i; y++) {
            std::vector<coord_t> row;
            for (size_t x = 0; x < i; x++) {
                row.push_back(0);
            }
            OM.push_back(row);
        }
        hom.push_back(OM);
    }
    
    // Construct depth buffer - 64x64
    buffer_t dbuf;
    for (size_t y = 0; y < 64; y++) {
        std::vector<std::shared_ptr<std::atomic<coord_t> > > row;
        for (size_t x = 0; x < 64; x++) {
            std::atomic<coord_t> *obj = new std::atomic<coord_t> (std::numeric_limits<coord_t>::infinity());
            row.push_back(std::shared_ptr<std::atomic<coord_t> > (obj));
        }
        dbuf.push_back(row);
    }

    coord_t max_z = -99;
    coord_t min_z = std::numeric_limits<coord_t>::infinity();
    Triangle tf;

    // Max and Min Z values
    for (auto &tri: tris) {
        coord_t temp = std::max({tri.v0.z, tri.v1.z, tri.v2.z});
        max_z = std::max({max_z, temp});
        temp = std::min({tri.v0.z, tri.v1.z, tri.v2.z});
        if (temp == -100) {
            tf = tri;
            continue;
        }
        min_z = std::min({min_z, temp});
    }
    
    omp_set_num_threads(num_threads);
    // This is the part we actually care about
    timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    if (hom_enabled) {
        float bucket_width = 0.2;

        // Init buckets
        std::vector<std::vector<Triangle>> buckets;
        for (coord_t i = min_z; i < max_z; i += bucket_width) {
            std::vector<Triangle> row;
            buckets.push_back(row);
        }

        // sort triangles into buckets 
        for (auto &tri : tris) {
            coord_t tmin = std::min({tri.v0.z, tri.v1.z, tri.v2.z});
            int bucket = (int)(floor((tmin - min_z)/bucket_width));

            if (tmin != -100)
                buckets[bucket].push_back(tri);
        }
        buckets[0].push_back(tf);
        
        int PO = 0;
        for (long unsigned int b = 0; b < buckets.size(); b++){
            #pragma omp parallel for
            for (auto &tri : buckets[b]) {
                if (!overlap_test(tri, hom, 4) || !depth_test(tri, dbuf)) {
                    rasterize_triangle(tri, zbuf, mode, hom[0], true);
                    update_dbuf(tri, dbuf, hom[0]);
                    PO++;
                    if (PO > 20) {
                        generate_hom_levels(hom);
                        PO = 0;
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for
        for (auto &tri : tris) {
            rasterize_triangle(tri, zbuf, mode, hom[0], false);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    size_t start_ns = (start_time.tv_sec*1000000000ull) + start_time.tv_nsec;
    size_t end_ns = (end_time.tv_sec*1000000000ull) + end_time.tv_nsec;
    size_t elapsed_ns = end_ns - start_ns;
    std::cout << "Elapsed time: " << (double)elapsed_ns/1000000 << "ms" << std::endl;

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
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <Input .tri> <Output .bmp> <X resolution> <Y resolution> <# threads> <raster mode 0-2> <hom enabled 0/1>" << std::endl;
        return 1;
    }

    // 1. Handle command-line params
    char *in_fname = argv[1];
    char *out_fname = argv[2];
    int x_res = atoi(argv[3]);
    int y_res = atoi(argv[4]);
    int num_threads = atoi(argv[5]);
    raster_mode_t mode = (raster_mode_t)atoi(argv[6]);
    bool hom_enabled = bool(atoi(argv[7]));

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
    buffer_t zbuf = rasterize_scene(tris, x_res, y_res, num_threads, mode, hom_enabled);
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

