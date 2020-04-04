//
// Created by jlebas01 on 04/04/2020.
//

#include <device/device.hpp>

__device__ float4 Hsv_FromRgbF(float r, float g, float b) {
    float M = 0.0, m = 0.0, c = 0.0;
    float4 HSVcolor = make_float4(0.f, 0.f, 0.f, 255.0f); //x : Hue, y : Saturation, z : Value, w : Opacity
    M = fmax(r, fmax(g, b));
    m = fmin(r, fmin(g, b));
    c = M - m;
    HSVcolor.z = M;
    if (c != 0.0f) {
        if (M == r) {
            HSVcolor.x = fmod(((g - b) / c), 6.0);
        } else if (M == g) {
            HSVcolor.x = (b - r) / c + 2.0;
        } else /*if(M==b)*/
        {
            HSVcolor.x = (r - g) / c + 4.0;
        }
        HSVcolor.x *= 60.0;
        HSVcolor.y = c / HSVcolor.z;
    }
    //}
    return HSVcolor;
}

__device__ float clip(float n, float lower, float upper) {
    return fmax(lower, fmin(n, upper));
}
