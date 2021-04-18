#include <inttypes.h>       // PRIu16
#include <stdint.h>         // uint16_t
#include <stdio.h>          // fclose, fopen, fprintf, fscanf, snprintf stderr
#include <string.h>         // strcmp

#include <memory>            // unique_ptr
#include <string>            // string, to_string
#include <type_traits>       // is_same
#include <vector>            // vector

#include <GLFW/glfw3.h>      // glfw*

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>       // stbi_image_free, stbi_load
#include <stb_image_write.h> // stbi_write_tga

#define TOOL_NAME "facewarp"

template <typename T>
static std::vector<T> parse_numberfile(const char * filename)
{
    static_assert(
        std::is_same<T, float   >::value ||
        std::is_same<T, uint16_t>::value,
        "Unsupported number type.");

    std::vector<T> out;

    const char * format = "";

    if (std::is_same<T, float   >::value) { format = "%f"; }
    if (std::is_same<T, uint16_t>::value) { format = "%" PRIu16; }

    std::unique_ptr<FILE, decltype(&fclose)> f(fopen(filename, "rb"), fclose);
    if (!f)
    {
        throw std::runtime_error("Failed to open file '" + std::string(filename) + "'.");
    }

    T number;
    while (fscanf(f.get(), format, &number) == 1)
    {
        out.push_back(number);
    }

    if (out.empty())
    {
        throw std::runtime_error("File '" + std::string(filename) + "' is either empty or in invalid format.");
    }

    return out;
}

struct image
{
    int                                                  width;
    int                                                  height;
    std::unique_ptr<uint8_t, decltype(&stbi_image_free)> data;

    image(const char * filename)
        : width (0)
        , height(0)
        , data  (nullptr, stbi_image_free)
    {
        int comp;
        data.reset(stbi_load(filename, &width, &height, &comp, 4));

        if (!data)
        {
            throw std::runtime_error("Failed to read image '" + std::string(filename) + "'.");
        }
    }
};

static int run(int argc, char ** argv)
{
    // Verify input.
    if (argc < 5 || argc > 8 || (argc == 2 && strcmp(argv[1], "--help") == 0))
    {
        fprintf(stderr,
            "USAGE:\n"
            "    " TOOL_NAME " image triangulation reference_points warped_points background_image [-novsync | -wireframe | -dump]\n"
            "\n"
            "OPTIONAL PARAMETERS:\n"
            "    -novsync    Disables the VSync. Faster processing, but not so nice to watch.\n"
            "    -wireframe  Renders a debug wireframe.\n"
            "    -dump       Dumps every processed frame to an image file with name format %%06i.tga.\n"
        );
        return 1;
    }

    // Parse optional arguments.
    bool no_vsync       = false;
    bool show_wireframe = false;
    bool dump_images    = false;

    for (int i = 6; i < argc; i++)
    {
        if      (strcmp(argv[i], "-novsync"  ) == 0) { no_vsync       = true; }
        else if (strcmp(argv[i], "-wireframe") == 0) { show_wireframe = true; }
        else if (strcmp(argv[i], "-dump"     ) == 0) { dump_images    = true; }
        else
        {
            throw std::runtime_error("Uncrecognized command line argument '" + std::string(argv[i]) + "'.");
        }
    }

    // Overwrite the PNG output settings.
    stbi_write_png_compression_level = 0;
    stbi__flip_vertically_on_write   = 1;

    // Parse the inputs.
    const image ref_img (argv[1]);
    const image ref_background_img (argv[5]);
    const auto  tri_idx = parse_numberfile<uint16_t>( argv[2]);
    const auto  ref_pts = parse_numberfile<float   >( argv[3]);
    const auto  wrp_pts = parse_numberfile<float   >( argv[4]);

    // Check that we have correct amount of warped points.
    if (wrp_pts.size() % ref_pts.size() != 0)
    {
        throw std::runtime_error("Warped points' count has to be multiple of reference points' count.");
    }

    // Create the OpenGL window.
    struct glfw_context
    {
         glfw_context() { if (glfwInit() != GLFW_TRUE) { throw std::runtime_error("GLFW initialization failed."); } }
        ~glfw_context() { glfwTerminate(); }
    }
    glfw;

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window(
        glfwCreateWindow(ref_img.width, ref_img.height, TOOL_NAME, nullptr, nullptr),
        glfwDestroyWindow
    );
    if (!window)
    {
        throw std::runtime_error("GLFW window creation failed");
    }

    glfwMakeContextCurrent(window.get());
    glfwSwapInterval(no_vsync ? 0 : 1);

    // Convert the image to premultiplied alpha.
    for (int i = 0, n = ref_img.width * ref_img.height * 4; i < n; i += 4)
    {
        uint8_t * pix = ref_img.data.get() + i;
        pix[0] = pix[0] * pix[3] / 255;
        pix[1] = pix[1] * pix[3] / 255;
        pix[2] = pix[2] * pix[3] / 255;
    }

    // Convert the image to premultiplied alpha.
    // for (int i = 0, n = ref_background_img.width * ref_background_img.height * 4; i < n; i += 4)
    // {
    //     uint8_t * pix = ref_background_img.data.get() + i;
    //     pix[0] = pix[0] / 255;
    //     pix[1] = pix[1] / 255;
    //     pix[2] = pix[2] / 255;
    // }

    // Upload the image to a texture texture.
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture  (GL_TEXTURE_2D, tex);
    glTexImage2D   (GL_TEXTURE_2D, 0, GL_RGBA, ref_img.width, ref_img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, ref_img.data.get());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


    GLuint bg_tex = 0;
    glGenTextures(1, &bg_tex);
    glBindTexture  (GL_TEXTURE_2D, bg_tex);
    glTexImage2D   (GL_TEXTURE_2D, 0, GL_RGBA, ref_background_img.width, ref_background_img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, ref_background_img.data.get());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


    // Process all the warped images.
    const size_t n = wrp_pts.size() / ref_pts.size();
          size_t i = 0;

    // Setup OpenGL state.
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glDisable(GL_DEPTH_TEST);
    // glEnable (GL_TEXTURE_2D);
    // glEnable (GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    // We're only using our single texture, so can bind it permanently.
    // glBindTexture(GL_TEXTURE_2D, tex);

    // Texture scaling factors.
    const float tex_scale_x = 1.0f / (float)(ref_img.width  - 1);
    const float tex_scale_y = 1.0f / (float)(ref_img.height - 1);

    // Output buffer for image dumping.
    std::vector<uint8_t> output_buffer;

    // Start processing.
    while (i < n && !glfwWindowShouldClose(window.get()) && glfwGetKey(window.get(), GLFW_KEY_ESCAPE) != GLFW_PRESS)
    {
        // Modify the window title to reflect the processed frame number.
        glfwSetWindowTitle(window.get(), (TOOL_NAME " - frame " + std::to_string(i + 1) + "/" + std::to_string(n)).c_str());

        int width, height;
        glfwGetFramebufferSize(window.get(), &width, &height);

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, ref_img.width, ref_img.height, 0, 0, 1);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();


        // Render current mesh.
        glEnable(GL_TEXTURE_2D);
        {
            glBindTexture(GL_TEXTURE_2D, bg_tex);
            glBegin(GL_QUADS);
            {
                glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

                glTexCoord2f(0, 0); glVertex2f(0, 0);
                glTexCoord2f(1, 0); glVertex2f(ref_img.width, 0); 
                glTexCoord2f(1, 1); glVertex2f(ref_img.width, ref_img.height); 
                glTexCoord2f(0, 1); glVertex2f(0, ref_img.height); 
            }
            glEnd();

            glEnable (GL_BLEND);
            glBindTexture(GL_TEXTURE_2D, tex);
            glBegin(GL_TRIANGLES);
            {
                glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

                // Points for this iteration.
                const auto * wrp_pts_i = wrp_pts.data() + i * ref_pts.size();

                // Render each 
                for (auto idx : tri_idx)
                {
                    const auto * ref_pt = ref_pts.data() + (ptrdiff_t)idx * 2;
                    const auto * wrp_pt = wrp_pts_i      + (ptrdiff_t)idx * 2;

                    glTexCoord2f(ref_pt[0] * tex_scale_x, ref_pt[1] * tex_scale_y);
                    glVertex2f  (wrp_pt[0]              , wrp_pt[1]              );
                }
            }
            glEnd();
            glDisable (GL_BLEND);



        }
        glDisable(GL_TEXTURE_2D);

        // Render the mesh itself.
        if (show_wireframe)
        {
            const float* wrp_pts_i = wrp_pts.data() + i * ref_pts.size();

            for (size_t j = 0; j < tri_idx.size(); j += 3)
            {
                glBegin(GL_LINE_LOOP);
                glColor4f(1.0f, 0.0f, 0.0f, 1.0f);

                for (size_t k = 0; k < 3; k++)
                {
                    const auto   idx = tri_idx[j + k];
                    const auto * pt  = wrp_pts_i + (ptrdiff_t)idx * 2;

                    glVertex2f(pt[0], pt[1]);
                }

                glEnd();
            }
        }

        // Dump the image.
        if (dump_images)
        {
            if (output_buffer.empty())
            {
                // Note that this is of the size of the default framebuffer size. For high-DPI devices,
                // the resolution will be larger than the size of the image, and would have to be scaled down.
                output_buffer.resize((size_t)width * (size_t)height * 4);
            }

            glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, output_buffer.data());

            // Convert the image to premultiplied alpha.
            // for (int i = 0, n = ref_background_img.width * ref_background_img.height * 4; i < n; i += 4)
            // {
            //     uint8_t * pix = output_buffer.data() + i;
            //     uint8_t * background_pix = ref_background_img.data.get() + i;
            //     pix[0] += background_pix[0];
            //     pix[1] += background_pix[1];
            //     pix[2] += background_pix[2];
            // }

            char filename[64];
            snprintf(filename, sizeof(filename), "%06zu.tga", i);

            if (stbi_write_tga(filename, width, height, 4, output_buffer.data()) == 0)
            {
                throw std::runtime_error("Saving of frame #" + std::to_string(i + 1) + " failed.");
            }
        }

        // Advance to the next frame.
        i++;

        glfwSwapBuffers(window.get());
        glfwPollEvents();
    }

    // Cleanup (all the rest is done via the unique_pointers).
    glDeleteTextures(1, &tex);

    return 0;
}

int main(int argc, char** argv)
{
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception & e)
    {
        fprintf(stderr, "EXCEPTION:\n    %s\n", e.what());
        return 1;
    }
}
