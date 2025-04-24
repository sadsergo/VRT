// Copyright 2020 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
#version 460

#extension GL_GOOGLE_include_directive : require
#include "../include/push_constants.h"
#include "../include/pbr.h"

#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_query : require

layout(local_size_x = WORKGROUP_WIDTH, local_size_y = WORKGROUP_HEIGHT, local_size_z = 1) in;

// The scalar layout qualifier here means to align types according to the alignment
// of their scalar components, instead of e.g. padding them to std140 rules.
layout(binding = 0, set = 0, scalar) buffer storageBuffer
{
  vec3 imageData[];
};

layout(binding = 1, set = 0) uniform accelerationStructureEXT tlas;

layout(binding = 2, set = 0, scalar) buffer Vertices
{
  vec3 vertices[];
};

layout(binding = 3, set = 0, scalar) buffer Indices
{
  uint indices[];
};

layout(push_constant) uniform PushConsts
{
  PushConstants pushConstants;
};

// Random number generation using pcg32i_random_t, using inc = 1. Our random state is a uint.
uint stepRNG(uint rngState)
{
  return rngState * 747796405 + 1;
}

// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
float stepAndOutputRNGFloat(inout uint rngState)
{
  // Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
  rngState  = stepRNG(rngState);
  uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
  word      = (word >> 22) ^ word;
  return float(word) / 4294967295.0f;
}

// Returns the color of the sky in a given direction (in linear color space)
vec3 skyColor(vec3 direction)
{
  // +y in world space is up, so:
  if(direction.y > 0.0f)
  {
    return mix(vec3(1.0f), vec3(0.25f, 0.5f, 1.0f), direction.y);
  }
  else
  {
    return vec3(0.03f);
  }
}

struct HitInfo
{
  vec3 color;
  vec3 worldPosition;
  vec3 worldNormal;
};

struct Light
{
  vec3 pos;
  vec3 dir;
  vec3 color;

  float angle;
  float intensity;
};

HitInfo getObjectHitInfo(rayQueryEXT rayQuery)
{
  HitInfo result;
  // Get the ID of the triangle
  const int primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);

  // Get the indices of the vertices of the triangle
  const uint i0 = indices[3 * primitiveID + 0];
  const uint i1 = indices[3 * primitiveID + 1];
  const uint i2 = indices[3 * primitiveID + 2];

  // Get the vertices of the triangle
  const vec3 v0 = vertices[i0];
  const vec3 v1 = vertices[i1];
  const vec3 v2 = vertices[i2];

  // Get the barycentric coordinates of the intersection
  vec3 barycentrics = vec3(0.0, rayQueryGetIntersectionBarycentricsEXT(rayQuery, true));
  barycentrics.x    = 1.0 - barycentrics.y - barycentrics.z;

  // Compute the coordinates of the intersection
  const vec3 objectPos = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
  // For the main tutorial, object space is the same as world space:
  result.worldPosition = objectPos;

  // Compute the normal of the triangle in object space, using the right-hand rule:
  //    v2      .
  //    |\      .
  //    | \     .
  //    |/ \    .
  //    /   \   .
  //   /|    \  .
  //  L v0---v1 .
  // n
  const vec3 objectNormal = normalize(cross(v1 - v0, v2 - v0));
  // For the main tutorial, object space is the same as world space:
  result.worldNormal = objectNormal;

  result.color = vec3(0.7f);

  return result;
}

void Intersect(rayQueryEXT rayQuery, const vec3 rayOrigin, const vec3 rayDirection)
{
  // Trace the ray and see if and where it intersects the scene!
  // First, initialize a ray query object:
  rayQueryInitializeEXT(rayQuery,              // Ray query
                        tlas,                  // Top-level acceleration structure
                        gl_RayFlagsOpaqueEXT,  // Ray flags, here saying "treat all geometry as opaque"
                        0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                        rayOrigin,             // Ray origin
                        0.0,                   // Minimum t-value
                        rayDirection,          // Ray direction
                        10000.0);              // Maximum t-value

  // Start traversal, and loop over all ray-scene intersections. When this finishes,
  // rayQuery stores a "committed" intersection, the closest intersection (if any).
  while(rayQueryProceedEXT(rayQuery))
  {
  }
}

void main()
{
  // The resolution of the buffer, which in this case is a hardcoded vector
  // of 2 unsigned integers:
  const uvec2 resolution = uvec2(pushConstants.render_width, pushConstants.render_height);

  Light lights[3];
  
  lights[0].pos = vec3(1.f, 2.f, 1.f);
  lights[0].color = vec3(1.f);
  lights[0].intensity = 1;

  lights[1].pos = vec3(2.f, 0.f, 0.f);
  lights[1].color = vec3(1.f);
  lights[1].intensity = 1;

  lights[2].pos = vec3(0.f, 1.f, 3.f);
  lights[2].color = vec3(1.f);
  lights[2].intensity = 1;

  // Get the coordinates of the pixel for this invocation:
  //
  // .-------.-> x
  // |       |
  // |       |
  // '-------'
  // v
  // y
  const uvec2 pixel = gl_GlobalInvocationID.xy;

  // If the pixel is outside of the image, don't do anything:
  if((pixel.x >= resolution.x) || (pixel.y >= resolution.y))
  {
    return;
  }

  // State of the random number generator.
  uint rngState = resolution.x * pixel.y + pixel.x;  // Initial seed

  // This scene uses a right-handed coordinate system like the OBJ file format, where the
  // +x axis points right, the +y axis points up, and the -z axis points into the screen.
  // The camera is located at (-0.001, 1, 6).
  const vec3 cameraOrigin = vec3(-0.001, 0, 3.0);
  // Define the field of view by the vertical slope of the topmost rays:
  const float fovVerticalSlope = 1.0 / 5.0;

  // The sum of the colors of all of the samples.
  vec3 summedPixelColor = vec3(0.0);

  // Limit the kernel to trace at most 64 samples.
  const int NUM_SAMPLES = 64;
  for(int sampleIdx = 0; sampleIdx < NUM_SAMPLES; sampleIdx++)
  {
    // Rays always originate at the camera for now. In the future, they'll
    // bounce around the scene.
    vec3 rayOrigin = cameraOrigin;
    // Compute the direction of the ray for this pixel. To do this, we first
    // transform the screen coordinates to look like this, where a is the
    // aspect ratio (width/height) of the screen:
    //           1
    //    .------+------.
    //    |      |      |
    // -a + ---- 0 ---- + a
    //    |      |      |
    //    '------+------'
    //          -1
    const vec2 randomPixelCenter = vec2(pixel) + vec2(stepAndOutputRNGFloat(rngState), stepAndOutputRNGFloat(rngState));
    const vec2 screenUV          = vec2((2.0 * randomPixelCenter.x - resolution.x) / resolution.y,    //
                               -(2.0 * randomPixelCenter.y - resolution.y) / resolution.y);  // Flip the y axis
    // Create a ray direction:
    vec3 rayDirection = vec3(fovVerticalSlope * screenUV.x, fovVerticalSlope * screenUV.y, -1.0);
    rayDirection      = normalize(rayDirection);

    vec3 accumulatedRayColor = vec3(1.0);  // The amount of light that made it to the end of the current ray.

    //  Get intersection info
    rayQueryEXT rayQuery;
    Intersect(rayQuery, rayOrigin, rayDirection);

    // Get the type of committed (true) intersection - nothing, a triangle, or
    // a generated object
    if(rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
    {
      // Ray hit a triangle
      HitInfo hitInfo = getObjectHitInfo(rayQuery);

      // Apply color absorption
      accumulatedRayColor *= hitInfo.color;

      // Start a new ray at the hit position, but offset it slightly along
      // the normal against rayDirection:
      rayOrigin = hitInfo.worldPosition - 0.0001 * sign(dot(rayDirection, hitInfo.worldNormal)) * hitInfo.worldNormal;

      vec3 summedLightColor = vec3(0.f);

      for (uint light_i = 0; light_i < 3; light_i++)
      {
        vec3 rayDirection_light = normalize(lights[light_i].pos - rayOrigin);

        rayQueryEXT rayQuery_light;
        Intersect(rayQuery_light, rayOrigin, rayDirection_light);

        if(rayQueryGetIntersectionTypeEXT(rayQuery_light, true) != gl_RayQueryCommittedIntersectionTriangleEXT)
        {
          summedLightColor += computePBR(hitInfo.worldNormal, -rayDirection, rayDirection_light, vec3(1.000, 0.766, 0.336), 1.f, 0.5, lights[light_i].color, dot(hitInfo.worldNormal, rayDirection_light)) * lights[light_i].intensity;
        }
      }

      summedLightColor = clamp(summedLightColor, 0.f, 1.f);
      summedPixelColor += summedLightColor;
    }
    else
    {
      summedPixelColor += vec3(1.f);
    }
  }

  // Get the index of this invocation in the buffer:
  uint linearIndex       = resolution.x * pixel.y + pixel.x;
  imageData[linearIndex] = summedPixelColor / NUM_SAMPLES;
}