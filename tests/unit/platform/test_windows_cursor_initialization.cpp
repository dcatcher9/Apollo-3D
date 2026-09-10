#include "src/platform/windows/display.h"

#include <gtest/gtest.h>

namespace {
  using namespace platf::dxgi;

  TEST(GpuCursorInitializationTest, ShapeBeforePositionHasDeterministicInvisibleViewport) {
    gpu_cursor_t cursor;
    EXPECT_FALSE(cursor.visible);
    EXPECT_EQ(cursor.display_rotation, DXGI_MODE_ROTATION_IDENTITY);
    EXPECT_EQ(cursor.texture_width, 0);
    EXPECT_EQ(cursor.texture_height, 0);
    EXPECT_EQ(cursor.display_width, 0);
    EXPECT_EQ(cursor.display_height, 0);
    EXPECT_FLOAT_EQ(cursor.cursor_view.MinDepth, 0);
    EXPECT_FLOAT_EQ(cursor.cursor_view.MaxDepth, 1);

    cursor.set_texture(32, 48, texture2d_t {});
    EXPECT_FALSE(cursor.visible);
    EXPECT_FLOAT_EQ(cursor.cursor_view.TopLeftX, 0);
    EXPECT_FLOAT_EQ(cursor.cursor_view.TopLeftY, 0);
    EXPECT_FLOAT_EQ(cursor.cursor_view.Width, 32);
    EXPECT_FLOAT_EQ(cursor.cursor_view.Height, 48);
  }

  TEST(GpuCursorInitializationTest, PositionBeforeShapeAndNegativeTopLeftRemainValid) {
    gpu_cursor_t cursor;
    cursor.set_pos(-12, 40, 1920, 1080, DXGI_MODE_ROTATION_IDENTITY, true);
    EXPECT_TRUE(cursor.visible);
    EXPECT_FLOAT_EQ(cursor.cursor_view.Width, 0);
    EXPECT_FLOAT_EQ(cursor.cursor_view.Height, 0);

    cursor.set_texture(32, 32, texture2d_t {});
    EXPECT_FLOAT_EQ(cursor.cursor_view.TopLeftX, -12);
    EXPECT_FLOAT_EQ(cursor.cursor_view.TopLeftY, 40);
    EXPECT_FLOAT_EQ(cursor.cursor_view.Width, 32);
    EXPECT_FLOAT_EQ(cursor.cursor_view.Height, 32);
  }

}  // namespace
