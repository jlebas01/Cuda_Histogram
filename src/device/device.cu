//
// Created by jlebas01 on 04/04/2020.
//

#include <devices/device.hpp>


namespace device {
    __device__ float4 fHSV_from_RGB(float r, float g, float b) {
        float M = 0.0f, m = 0.0f, c = 0.0f;
        float4 HSVcolor = make_float4(0.f, 0.f, 0.f, 255.0f); //x : Hue, y : Saturation, z : Value, w : Opacity
        M = fmax(r, fmax(g, b));
        m = fmin(r, fmin(g, b));
        c = M - m;
        HSVcolor.z = M;
        if (c != 0.0f) {
            if (M == r) {
                HSVcolor.x = fmod(((g - b) / c), 6.0f);
            } else if (M == g) {
                HSVcolor.x = (b - r) / c + 2.0f;
            } else /*if(M==b)*/
            {
                HSVcolor.x = (r - g) / c + 4.0f;
            }
            HSVcolor.x *= 60.0f;
            HSVcolor.y = c / HSVcolor.z;
        }
        //}
        return HSVcolor;
    }

    __device__ float4 fRGB_from_HSV(float h, float s, float v) {
        float c = 0.0f, m = 0.0f, x = 0.0f;
        float4 color = make_float4(0.f, 0.f, 0.f, 1.0f);
       // if (Hsv_IsValid(h, s, v) == true) {
            c = v * s;
            x = c * (1.0f - fabs(fmod(h / 60.0f, 2.0f) - 1.0f));
            m = v - c;
            if (h >= 0.0f && h < 60.0) {
                color = make_float4(c + m, x + m, m, 1.0f);
            } else if (h >= 60.0f && h < 120.0f) {
                color = make_float4(x + m, c + m, m, 1.0f);
            } else if (h >= 120.0f && h < 180.0f) {
                color = make_float4(m, c + m, x + m, 1.0f);
            } else if (h >= 180.0f && h < 240.0f) {
                color = make_float4(m, x + m, c + m, 1.0f);
            } else if (h >= 240.0f && h < 300.0f) {
                color = make_float4(x + m, m, c + m, 1.0f);
            } else if (h >= 300.0f && h < 360.0f) {
                color = make_float4(c + m, m, x + m, 1.0f);
            } else {
                color = make_float4(m, m, m, 1.0f);
            }
      //  }
        return color;
    }

    __device__ float clip(float n, float lower, float upper) {
        return fmax(lower, fmin(n, upper));
    }

    __device__ float4 normalizeRGB(float r, float g, float b, float o) {
        float4 RGBColorNormalized = make_float4(r / 255.f, g / 255.f, b / 255.f, o / 255.f);
        return RGBColorNormalized;
    }
}
