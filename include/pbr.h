#ifndef PBR_H
#define PBR_H

#pragma once

#define M_PI 3.1415926535897932384626433832795f

#ifdef __cplusplus
#include <glm/glm.hpp>
using namespace glm;
#endif

// struct Light
// {
//     vec3 pos;
//     vec3 dir;
//     vec3 color;

//     float angle;
//     float intensity;
// };

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (M_PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0f);
    float NdotL = max(dot(N, L), 0.0f);
    float ggx1 = GeometrySchlickGGX(NdotV, roughness);
    float ggx2 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

// Final PBR BRDF function
vec3 computePBR(
    vec3 N, vec3 V, vec3 L,             // normal, view, light directions (all normalized)
    vec3 albedo, float metallic, float roughness,
    vec3 lightColor, float NdotL        // precomputed dot(N, L) if available
) {
    vec3 H = normalize(V + L);
    float NdotV = max(dot(N, V), 0.0f);
    float NdotH = max(dot(N, H), 0.0f);
    float HdotV = max(dot(H, V), 0.0f);

    // Base reflectivity
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnelSchlick(HdotV, F0);
    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);

    vec3 numerator = D * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.001;
    vec3 specular = numerator / denominator;

    // Energy conservation: only non-metals get diffuse
    vec3 kS = F;
    vec3 kD = vec3(1.0f) - kS;
    kD *= 1.0f - metallic;

    vec3 diffuse = kD * albedo / M_PI;

    return (diffuse + specular) * lightColor * NdotL;
}

#endif