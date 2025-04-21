#ifndef UNIFORM_BUFFS_H
#define UNIFORM_BUFFS_H

#pragma once

#define WORKGROUP_WIDTH 16
#define WORKGROUP_HEIGHT 8

#ifdef __cplusplus
#include <cstdint>

using uint = uint32_t;
#endif

struct PushConstants
{
  uint render_width;
  uint render_height;
};

#endif // UNIFORM_BUFFS_H