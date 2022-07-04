#include "LocalWarp.h"

//在能量函数比较那里用到的重新定义的比较方法
bool cmp(const pair<int, float> a, const pair<int, float> b) {
    return a.second < b.second;
}

bool cmpd(const pair<int, double> a, const pair<int, double> b) {
    return a.second < b.second;
}

//判断是否为透明点，如果是黑色那么返回false，如果是空白就是true
bool isTransparent(CVMat mask, int row, int col) {
    if (mask.at<uchar>(row, col) == 0) {
        return false;
    }
    else {
        return  true;
    }
}



//初始化偏移场U
void initDisplacement(vector<vector<CoordinateInt>>& displacement, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        vector<CoordinateInt> displacement_row;
        for (int col = 0; col < cols; col++) {
            CoordinateInt c;
            displacement_row.push_back(c);
        }
        displacement.push_back(displacement_row);
    }
}



//计算边缘能量，利用Sobel算子计算整张图的能量，Origin Seam carving用的能量函数
CVMat calcEnergy(CVMat src) {
    CVMat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    CVMat grad_x, grad_y, dst;
    //CVMat abs_grad_x, abs_grad_y;
    //边缘检测算子，利用像素点的梯度计算物体边缘
    cv::Sobel(gray, grad_x, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(gray, grad_y, CV_8U, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    //cv::convertScaleAbs(grad_x, abs_grad_x);
    //cv::convertScaleAbs(grad_y, abs_grad_y);
    cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, dst);
    return dst;
}


//查找上下左右边缘四条边的最长的连续空白边
pair<int, int> chooseLongestBorder(CVMat src, CVMat mask, BORDER& direction) {
    int rows = src.rows;
    int cols = src.cols;
    //记录着最长的空白边长度，和开始、结束的位置
    int maxL = 0;
    int final_beginIdx = 0;
    int final_endIdx = 0;

    //LEFT
    int tmp_maxL, tmp_beginIdx, tmp_endIdx;
    tmp_maxL = tmp_beginIdx = tmp_endIdx = 0;
    bool isCounting = false;
    for (int row = 0; row < rows; row++) {
        if (!isTransparent(mask, row, 0) || row == rows - 1) {//如遇黑点或者尽头
            if (isCounting) {
                if (isTransparent(mask, row, 0)) {
                    tmp_endIdx++;
                    tmp_maxL++;
                    isCounting = true;
                }
                if (tmp_maxL > maxL) {
                    maxL = tmp_maxL;
                    final_beginIdx = tmp_beginIdx;
                    final_endIdx = tmp_endIdx;
                    direction = BORDER_LEFT;
                }
            }
            isCounting = false;
            tmp_beginIdx = tmp_endIdx = row + 1;
            tmp_maxL = 0;
        }
        else { //该点空白，开始计数
            tmp_endIdx++;
            tmp_maxL++;
            isCounting = true;
        }
    }

    //RIGHT
    tmp_maxL = tmp_beginIdx = tmp_endIdx = 0;
    isCounting = false;
    for (int row = 0; row < rows; row++) {
        if (!isTransparent(mask, row, cols - 1) || row == rows - 1) {
            if (isCounting) {
                if (isTransparent(mask, row, cols - 1)) {
                    tmp_endIdx++;
                    tmp_maxL++;
                    isCounting = true;
                }
                if (tmp_maxL > maxL) {
                    maxL = tmp_maxL;
                    final_beginIdx = tmp_beginIdx;
                    final_endIdx = tmp_endIdx;
                    direction = BORDER_RIGHT;
                }
            }
            isCounting = false;
            tmp_beginIdx = tmp_endIdx = row + 1;
            tmp_maxL = 0;
        }
        else {
            tmp_endIdx++;
            tmp_maxL++;
            isCounting = true;
        }
    }

    //TOP
    tmp_maxL = tmp_beginIdx = tmp_endIdx = 0;
    isCounting = false;
    for (int col = 0; col < cols; col++) {
        if (!isTransparent(mask, 0, col) || col == cols - 1) {
            if (isCounting) {
                if (isTransparent(mask, 0, col)) {
                    tmp_endIdx++;
                    tmp_maxL++;
                    isCounting = true;
                }
                if (tmp_maxL > maxL) {
                    maxL = tmp_maxL;
                    final_beginIdx = tmp_beginIdx;
                    final_endIdx = tmp_endIdx;
                    direction = BORDER_TOP;
                }
            }
            isCounting = false;
            tmp_beginIdx = tmp_endIdx = col + 1;
            tmp_maxL = 0;
        }
        else {
            tmp_endIdx++;
            tmp_maxL++;
            isCounting = true;
        }
    }

    //BOTTOM
    tmp_maxL = tmp_beginIdx = tmp_endIdx = 0;
    isCounting = false;
    for (int col = 0; col < cols; col++) {
        if (!isTransparent(mask, rows - 1, col) || col == cols - 1) {
            if (isCounting) {
                if (isTransparent(mask, rows - 1, col)) {
                    tmp_endIdx++;
                    tmp_maxL++;
                    isCounting = true;
                }
                if (tmp_maxL > maxL) {
                    maxL = tmp_maxL;
                    final_beginIdx = tmp_beginIdx;
                    final_endIdx = tmp_endIdx;
                    direction = BORDER_BOTTOM;
                }
            }
            isCounting = false;
            tmp_beginIdx = tmp_endIdx = col + 1;
            tmp_maxL = 0;
        }
        else {
            tmp_endIdx++;
            tmp_maxL++;
            isCounting = true;
        }
    }

    if (maxL == 0) {
        return make_pair(0, 0);
    }
    else {
        return make_pair(final_beginIdx, final_endIdx - 1);
    }
}


//进行局部的Seam Carving算法，同时得到最终的偏移量矩阵
vector<vector<CoordinateInt>> localWarp(CVMat src, CVMat& warp_img, CVMat mask) {

    vector<vector<CoordinateInt>> displacementMap = getLocalWarpDisplacement(src, mask);
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            CoordinateInt displacement = displacementMap[row][col];
            colorPixel pixel = src.at<colorPixel>(row + displacement.row, col + displacement.col);
            warp_img.at<colorPixel>(row, col) = pixel;
        }
    }
    return displacementMap;
}



//DP找到能量最低的Seam,基于Forward Energy
int* getLocalSeam_improved(CVMat src, CVMat mask, SEAMDIRECTION seamDirection, pair<int, int> begin_end) {
    if (seamDirection == SEAM_HORIZONTAL) {
        cv::transpose(src, src);
        cv::transpose(mask, mask);
    }

    int rows = src.rows;
    int cols = src.cols;

    int row_begin = begin_end.first;
    int row_end = begin_end.second;

    int range = row_end - row_begin + 1;

    int col_begin = 0;
    int col_end = cols - 1;

    int outputWidth = cols;
    int outputHeight = range;

    CVMat displayimg;
    src.copyTo(displayimg);
    //Range左闭右开
    CVMat local_img = displayimg(cv::Range::Range(row_begin, row_end + 1), cv::Range::Range(col_begin, col_end + 1));
    CVMat local_mask = mask(cv::Range::Range(row_begin, row_end + 1), cv::Range::Range(col_begin, col_end + 1));
    //根据forward energy function算子计算局部图的能量
    //CVMat local_energy = forwardEnergy(local_img, local_mask, col_begin, col_end);
    CVMat gray;
    cv::cvtColor(local_img, gray, cv::COLOR_BGR2GRAY);

    //计算每一个像素点的forward energy
    CVMat C_U(gray.rows, gray.cols, CV_64F, cv::Scalar(0)), C_L(gray.rows, gray.cols, CV_64F, cv::Scalar(0)), C_R(gray.rows, gray.cols, CV_64F, cv::Scalar(0));
    for (int y = 0; y < range; y++) {
        for (int x = 0; x < cols; x++) {
            int xl = (x > 0) ? (x - 1) : x;
            int xr = (x < cols - 1) ? (x + 1) : x;

            double current_diff = abs(gray.at<uchar>(y, xr) - gray.at<uchar>(y, xl));
            C_U.at<double>(y, x) = current_diff;
            C_L.at<double>(y, x) = (y > 0) ? current_diff + abs(gray.at<uchar>(y - 1, x) - gray.at<uchar>(y, xl)) : 0;
            C_R.at<double>(y, x) = (y > 0) ? current_diff + abs(gray.at<uchar>(y - 1, x) - gray.at<uchar>(y, xr)) : 0;
        }
    }

    CVMat energyMMap(range, cols, CV_64F, cv::Scalar(0));
    //将每一个空的像素点能量设置为INF
    for (int row = 0; row < range; row++) {
        for (int col = col_begin; col <= col_end; col++) {
            if ((int)local_mask.at<uchar>(row, col) == 255) {
                energyMMap.at<double>(row, col) = INF;
            }
        }
    }
    //计算累积能量
    for (int row = 1; row < range; row++) {
        for (int col = col_begin; col <= col_end; col++) {
            if (col == col_begin) {
                energyMMap.at<double>(row, col) += min(energyMMap.at<double>(row - 1, col) + C_U.at<double>(row, col), energyMMap.at<double>(row - 1, col + 1) + C_R.at<double>(row, col));
            }
            else {
                if (col == col_end) {
                    energyMMap.at<double>(row, col) += min(energyMMap.at<double>(row - 1, col) + C_U.at<double>(row, col), energyMMap.at<double>(row - 1, col - 1) + C_L.at<double>(row, col));
                }
                else {
                    energyMMap.at<double>(row, col) += min(energyMMap.at<double>(row - 1, col) + C_U.at<double>(row, col), min(energyMMap.at<double>(row - 1, col - 1) + C_L.at<double>(row, col), energyMMap.at<double>(row - 1, col + 1) + C_R.at<double>(row, col)));
                }
            }
        }
    }
    CVMat tmpEnergy; //累积能量矩阵
    energyMMap.copyTo(tmpEnergy);
    //利用动态规划算法获取最小能量线
    //获取最后一行的能量值
    vector<pair<int, double>> last_row;
    for (int col = col_begin; col <= col_end; col++) {
        last_row.push_back(make_pair(col, tmpEnergy.at<double>(range - 1, col)));
    }
    //对最后一行的能量值进行排序
    sort(last_row.begin(), last_row.end(), cmpd);
    int* seam = new int[range];
    //得到能量线在最后一行的像素
    seam[range - 1] = last_row[0].first;
    //逆序往前，找能量线的像素，只需要逆序找最小的就行了
    for (int row = range - 2; row >= 0; row--) {
        if (seam[row + 1] == col_begin) {
            if (tmpEnergy.at<double>(row, seam[row + 1] + 1) < tmpEnergy.at<double>(row, seam[row + 1])) {
                seam[row] = seam[row + 1] + 1;
            }
            else {
                seam[row] = seam[row + 1];
            }
        }
        else {
            if (seam[row + 1] == col_end) {
                if (tmpEnergy.at<double>(row, seam[row + 1] - 1) < tmpEnergy.at<double>(row, seam[row + 1])) {
                    seam[row] = seam[row + 1] - 1;
                }
                else {
                    seam[row] = seam[row + 1];
                }
            }
            else {
                double min_energy = min(tmpEnergy.at<double>(row, seam[row + 1] - 1), min(tmpEnergy.at<double>(row, seam[row + 1]), tmpEnergy.at<double>(row, seam[row + 1] + 1)));
                if (min_energy == tmpEnergy.at<double>(row, seam[row + 1] - 1)) {
                    seam[row] = seam[row + 1] - 1;
                }
                else if (min_energy == tmpEnergy.at<double>(row, seam[row + 1])) {
                    seam[row] = seam[row + 1];
                }
                else {
                    seam[row] = seam[row + 1] + 1;
                }
            }
        }
    }
    return seam;
}


//将seam插入图片当中去
CVMat insertLocalSeam(CVMat src, CVMat& mask, int* seam, SEAMDIRECTION seamDirection, pair<int, int> begin_end, bool shift2end) {
    //先进行方向的转换，都看成竖的能量线来处理
    if (seamDirection == SEAM_HORIZONTAL) {
        cv::transpose(src, src);
        cv::transpose(mask, mask);
    }

    CVMat resimg;
    src.copyTo(resimg);

    int begin = begin_end.first;
    int end = begin_end.second;
    int rows = src.rows;
    int cols = src.cols;

    for (int row = begin; row <= end; row++) {
        int local_row = row - begin;
        if (!shift2end) {
            //对空缺在左边或者上边的细缝，将seam左边的像素向左平移一位
            for (int col = 0; col < seam[local_row]; col++) {
                colorPixel p = src.at<colorPixel>(row, col + 1);
                resimg.at<colorPixel>(row, col) = p;
                mask.at<uchar>(row, col) = mask.at<uchar>(row, col + 1);
            }
        }
        else {
            //对空缺在右边或者下边的细缝，将seam右边的像素向右平移一位
            for (int col = cols - 1; col > seam[local_row]; col--) {
                colorPixel p = src.at<colorPixel>(row, col - 1);
                resimg.at<colorPixel>(row, col) = p;
                mask.at<uchar>(row, col) = mask.at<uchar>(row, col - 1);
            }
        }
        //对seam细缝赋值上色；
        mask.at<uchar>(row, seam[local_row]) = 0;
        if (seam[local_row] == 0) {
            resimg.at<colorPixel>(row, seam[local_row]) = src.at<colorPixel>(row, seam[local_row] + 1);
        }
        else {
            if (seam[local_row] == cols - 1) {
                resimg.at<colorPixel>(row, seam[local_row]) = src.at<colorPixel>(row, seam[local_row] - 1);
            }
            else {
                colorPixel p1 = src.at<colorPixel>(row, seam[local_row] - 1);
                colorPixel p2 = src.at<colorPixel>(row, seam[local_row] + 1);
                resimg.at<colorPixel>(row, seam[local_row]) = 0.5 * p1 + 0.5 * p2;
            }
        }
    }
    if (seamDirection == SEAM_HORIZONTAL) {
        cv::transpose(resimg, resimg);
        cv::transpose(mask, mask);
    }
    return resimg;
}



//DP找到能量最低的Seam，基于backward能量
int* getLocalSeam(CVMat src, CVMat mask, SEAMDIRECTION seamDirection, pair<int, int> begin_end) {
    //统一寻找竖直的seam
    if (seamDirection == SEAM_HORIZONTAL) {
        cv::transpose(src, src);
        cv::transpose(mask, mask);
    }

    int rows = src.rows;
    int cols = src.cols;

    int row_begin = begin_end.first;
    int row_end = begin_end.second;

    int range = row_end - row_begin + 1;

    int col_begin = 0;
    int col_end = cols - 1;

    int outputWidth = cols;
    int outputHeight = range;

    CVMat displayimg;
    src.copyTo(displayimg);
    //Range左闭右开
    CVMat local_img = displayimg(cv::Range::Range(row_begin, row_end + 1), cv::Range::Range(col_begin, col_end + 1));
    CVMat local_mask = mask(cv::Range::Range(row_begin, row_end + 1), cv::Range::Range(col_begin, col_end + 1));
    //根据Sobel算子计算局部图的能量
    CVMat local_energy = calcEnergy(local_img);
    CVMat local_energy_32f;
    local_energy.convertTo(local_energy_32f, CV_32F);
    //将缺失部分的能量设置为无穷
    for (int row = 0; row < range; row++) {
        for (int col = col_begin; col <= col_end; col++) {
            if ((int)local_mask.at<uchar>(row, col) == 255) {
                local_energy_32f.at<float>(row, col) = INF;
            }
        }
    }
    CVMat tmpEnergy;
    local_energy_32f.copyTo(tmpEnergy);
    //利用动态规划算法获取最小能量线
    //求累计能量矩阵
    for (int row = 1; row < range; row++) {
        for (int col = col_begin; col <= col_end; col++) {
            if (col == col_begin) {
                tmpEnergy.at<float>(row, col) += min(tmpEnergy.at<float>(row - 1, col), tmpEnergy.at<float>(row - 1, col + 1));
            }
            else {
                if (col == col_end) {
                    tmpEnergy.at<float>(row, col) += min(tmpEnergy.at<float>(row - 1, col), tmpEnergy.at<float>(row - 1, col - 1));
                }
                else {
                    tmpEnergy.at<float>(row, col) += min(tmpEnergy.at<float>(row - 1, col), min(tmpEnergy.at<float>(row - 1, col + 1), tmpEnergy.at<float>(row - 1, col - 1)));
                }
            }
        }
    }
    //获取最后一行的能量值
    vector<pair<int, float>> last_row;
    for (int col = col_begin; col <= col_end; col++) {
        last_row.push_back(make_pair(col, tmpEnergy.at<float>(range - 1, col)));
    }
    //对最后一行的能量值进行排序
    sort(last_row.begin(), last_row.end(), cmp);
    int* seam = new int[range];
    //得到能量线在最后一行的像素
    seam[range - 1] = last_row[0].first;
    //逆序往前，找能量线的像素，只需要逆序找最小的就行了
    for (int row = range - 2; row >= 0; row--) {
        if (seam[row + 1] == col_begin) {
            if (tmpEnergy.at<float>(row, seam[row + 1] + 1) < tmpEnergy.at<float>(row, seam[row + 1])) {
                seam[row] = seam[row + 1] + 1;
            }
            else {
                seam[row] = seam[row + 1];
            }
        }
        else {
            if (seam[row + 1] == col_end) {
                if (tmpEnergy.at<float>(row, seam[row + 1] - 1) < tmpEnergy.at<float>(row, seam[row + 1])) {
                    seam[row] = seam[row + 1] - 1;
                }
                else {
                    seam[row] = seam[row + 1];
                }
            }
            else {
                float min_energy = min(tmpEnergy.at<float>(row, seam[row + 1] - 1), min(tmpEnergy.at<float>(row, seam[row + 1]), tmpEnergy.at<float>(row, seam[row + 1] + 1)));
                if (min_energy == tmpEnergy.at<float>(row, seam[row + 1] - 1)) {
                    seam[row] = seam[row + 1] - 1;
                }
                else {
                    if (min_energy == tmpEnergy.at<float>(row, seam[row + 1])) {
                        seam[row] = seam[row + 1];
                    }
                    else {
                        seam[row] = seam[row + 1] + 1;
                    }
                }
            }
        }
    }
    return seam;
}



//确定wrapped之后的网格点
vector<vector<CoordinateDouble>> getRectangleMesh(CVMat src, Config config) {
    int meshNumRow = config.meshNumRow;
    int meshNumCol = config.meshNumCol;
    double rowPerMesh = config.rowPerMesh;
    double colPerMesh = config.colPerMesh;

    vector<vector<CoordinateDouble>> mesh;
    for (int row_mesh = 0; row_mesh < meshNumRow; row_mesh++) {
        vector<CoordinateDouble> meshRow;
        for (int col_mesh = 0; col_mesh < meshNumCol; col_mesh++) {
            CoordinateDouble coord;
            coord.row = row_mesh * rowPerMesh;
            coord.col = col_mesh * colPerMesh;
            meshRow.push_back(coord);
        }
        mesh.push_back(meshRow);
    }
    return mesh;
}



//warp back！！将规整的网格点按照偏移场还原回原图
void warpMeshBack(vector<vector<CoordinateDouble>>& mesh, vector<vector<CoordinateInt>> displacementMap, Config config) {
    int meshNumRow = config.meshNumRow;
    int meshNumCol = config.meshNumCol;

    for (int row_mesh = 0; row_mesh < meshNumRow; row_mesh++) {
        for (int col_mesh = 0; col_mesh < meshNumCol; col_mesh++) {
            if (row_mesh == meshNumRow - 1 && col_mesh == meshNumCol - 1) {
                CoordinateDouble& meshVertexCoord = mesh[row_mesh][col_mesh];
                CoordinateInt vertexDisplacement = displacementMap[floor(meshVertexCoord.row) - 1][floor(meshVertexCoord.col) - 1];
                meshVertexCoord.row += vertexDisplacement.row;
                meshVertexCoord.col += vertexDisplacement.col;
            }
            CoordinateDouble& meshVertexCoord = mesh[row_mesh][col_mesh];
            CoordinateInt vertexDisplacement = displacementMap[(int)floor(meshVertexCoord.row)][(int)floor(meshVertexCoord.col)];
            meshVertexCoord.row += vertexDisplacement.row;
            meshVertexCoord.col += vertexDisplacement.col;
        }
    }
}




//获取偏移量矩阵U
vector<vector<CoordinateInt>> getLocalWarpDisplacement(CVMat src, CVMat mask) {

    int rows = src.rows;
    int cols = src.cols;
    //记录偏移数组容器
    vector<vector<CoordinateInt>> displacementMap; //最终的
    vector<vector<CoordinateInt>> finalDisplacementMap; //临时更新
    //初始化容器
    initDisplacement(finalDisplacementMap, rows, cols);
    initDisplacement(displacementMap, rows, cols);

    while (true) {
        BORDER direction;
        pair<int, int> begin_end = chooseLongestBorder(src, mask, direction);
        //cout << direction << endl;
        if (begin_end.first == 0 && begin_end.second == 0) {
            //cv::imwrite("seam_img.png", seam_img);
            return displacementMap;
        }
        else {
            bool shift2end = false;
            SEAMDIRECTION seamdirection;
            switch (direction) {
            case BORDER_LEFT:
                seamdirection = SEAM_VERTICAL;
                shift2end = false;
                break;
            case BORDER_RIGHT:
                seamdirection = SEAM_VERTICAL;
                shift2end = true;
                break;
            case BORDER_TOP:
                seamdirection = SEAM_HORIZONTAL;
                shift2end = false;
                break;
            case BORDER_BOTTOM:
                seamdirection = SEAM_HORIZONTAL;
                shift2end = true;
                break;
            default:
                break;
            }
            //通过最小化能量函数，获取一条最小能量线
            //int* seam = getLocalSeam(src, mask, seamdirection, begin_end);
            int* seam = getLocalSeam_improved(src, mask, seamdirection, begin_end);

            //插入最小能量线，同时对需要位移的像素进行位移
            src = insertLocalSeam(src, mask, seam, seamdirection, begin_end, shift2end);

            //更新偏移矩阵
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    CoordinateInt tmpdisplacement;
                    if (seamdirection == SEAM_VERTICAL && row >= begin_end.first && row <= begin_end.second) {
                        int local_row = row - begin_end.first;
                        if (col > seam[local_row] && shift2end) {//右移
                            tmpdisplacement.col = -1;
                        }
                        else {
                            if (col < seam[local_row] && !shift2end) { //左移
                                tmpdisplacement.col = 1;
                            }
                        }
                    }
                    else {
                        if (seamdirection == SEAM_HORIZONTAL && col >= begin_end.first && col <= begin_end.second) {
                            int local_col = col - begin_end.first;
                            if (row > seam[local_col] && shift2end) { //下移
                                tmpdisplacement.row = -1;
                            }
                            else {
                                if (row < seam[local_col] && !shift2end) { //上移
                                    tmpdisplacement.row = 1;
                                }
                            }
                        }
                    }
                    //准备记录偏移量
                    CoordinateInt& finalDisplacement = finalDisplacementMap[row][col];
                    //获得上一时刻的坐标
                    int tmpDisplacement_row = row + tmpdisplacement.row;
                    int tmpDisplacement_col = col + tmpdisplacement.col;
                    //利用上一时刻的坐标在displacementMap上对应点获取上一时刻的偏移量
                    CoordinateInt displacementOfTarget = displacementMap[tmpDisplacement_row][tmpDisplacement_col];
                    //利用上一时刻坐标和偏移量计算出起始点
                    int rowOfOrigin = tmpDisplacement_row + displacementOfTarget.row;
                    int colOfOrigin = tmpDisplacement_col + displacementOfTarget.col;
                    //利用起始点和现在的坐标相减，获得现在的偏移量
                    finalDisplacement.row = rowOfOrigin - row;
                    finalDisplacement.col = colOfOrigin - col;
                }
            }
            //将finalDisplacementMap复制给displacementMap然后进行下一次扩张！
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    CoordinateInt& displacement = displacementMap[row][col];
                    CoordinateInt finalDisplacement = finalDisplacementMap[row][col];
                    displacement.row = finalDisplacement.row;
                    displacement.col = finalDisplacement.col;
                }
            }
        }
    }
    return displacementMap;
}