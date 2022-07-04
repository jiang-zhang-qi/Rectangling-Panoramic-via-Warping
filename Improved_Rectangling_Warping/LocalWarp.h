#pragma once
#ifndef LocalWarp_hpp
#define LocalWarp_hpp
#include "ComLib.h"

enum BORDER {
    BORDER_TOP = 0,
    BORDER_BOTTOM = 1,
    BORDER_LEFT = 2,
    BORDER_RIGHT = 3
};

enum SEAMDIRECTION {
    SEAM_VERTICAL = 0,
    SEAM_HORIZONTAL = 1
};

vector<vector<CoordinateInt>> localWarp(const CVMat src, CVMat& warp_img, CVMat mask);
//CVMat insertLocalSeam(CVMat src, CVMat& seam_img, CVMat& mask, int* seam, SEAMDIRECTION seamDirection, pair<int, int> begin_end, bool shift2end);
CVMat insertLocalSeam(CVMat src, CVMat& mask, int* seam, SEAMDIRECTION seamDirection, pair<int, int> begin_end, bool shift2end);

int* getLocalSeam(CVMat src, CVMat mask, SEAMDIRECTION seamDirection, pair<int, int> begin_end);
int* getLocalSeam_improved(CVMat src, CVMat mask, SEAMDIRECTION seamDirection, pair<int, int> begin_end);

//vector<vector<CoordinateInt>> getLocalWarpDisplacement(CVMat& warp_img, CVMat mask);
vector<vector<CoordinateInt>> getLocalWarpDisplacement(CVMat src, CVMat mask);
pair<int, int> chooseLongestBorder(CVMat src, CVMat mask, BORDER& direction);


void warpMeshBack(vector<vector<CoordinateDouble>>& mesh, vector<vector<CoordinateInt>> displacementMap, Config config);
vector<vector<CoordinateDouble>> getRectangleMesh(CVMat src, Config config);

#endif // !LocalWarp_hpp
#pragma once
