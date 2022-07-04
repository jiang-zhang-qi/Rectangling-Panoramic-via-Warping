#include "LocalWarp.h"

//�����������Ƚ������õ������¶���ıȽϷ���
bool cmp(const pair<int, float> a, const pair<int, float> b) {
    return a.second < b.second;
}

bool cmpd(const pair<int, double> a, const pair<int, double> b) {
    return a.second < b.second;
}

//�ж��Ƿ�Ϊ͸���㣬����Ǻ�ɫ��ô����false������ǿհ׾���true
bool isTransparent(CVMat mask, int row, int col) {
    if (mask.at<uchar>(row, col) == 0) {
        return false;
    }
    else {
        return  true;
    }
}



//��ʼ��ƫ�Ƴ�U
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



//�����Ե����������Sobel���Ӽ�������ͼ��������Origin Seam carving�õ���������
CVMat calcEnergy(CVMat src) {
    CVMat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    CVMat grad_x, grad_y, dst;
    //CVMat abs_grad_x, abs_grad_y;
    //��Ե������ӣ��������ص���ݶȼ��������Ե
    cv::Sobel(gray, grad_x, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(gray, grad_y, CV_8U, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    //cv::convertScaleAbs(grad_x, abs_grad_x);
    //cv::convertScaleAbs(grad_y, abs_grad_y);
    cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, dst);
    return dst;
}


//�����������ұ�Ե�����ߵ���������հױ�
pair<int, int> chooseLongestBorder(CVMat src, CVMat mask, BORDER& direction) {
    int rows = src.rows;
    int cols = src.cols;
    //��¼����Ŀհױ߳��ȣ��Ϳ�ʼ��������λ��
    int maxL = 0;
    int final_beginIdx = 0;
    int final_endIdx = 0;

    //LEFT
    int tmp_maxL, tmp_beginIdx, tmp_endIdx;
    tmp_maxL = tmp_beginIdx = tmp_endIdx = 0;
    bool isCounting = false;
    for (int row = 0; row < rows; row++) {
        if (!isTransparent(mask, row, 0) || row == rows - 1) {//�����ڵ���߾�ͷ
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
        else { //�õ�հף���ʼ����
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


//���оֲ���Seam Carving�㷨��ͬʱ�õ����յ�ƫ��������
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



//DP�ҵ�������͵�Seam,����Forward Energy
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
    //Range����ҿ�
    CVMat local_img = displayimg(cv::Range::Range(row_begin, row_end + 1), cv::Range::Range(col_begin, col_end + 1));
    CVMat local_mask = mask(cv::Range::Range(row_begin, row_end + 1), cv::Range::Range(col_begin, col_end + 1));
    //����forward energy function���Ӽ���ֲ�ͼ������
    //CVMat local_energy = forwardEnergy(local_img, local_mask, col_begin, col_end);
    CVMat gray;
    cv::cvtColor(local_img, gray, cv::COLOR_BGR2GRAY);

    //����ÿһ�����ص��forward energy
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
    //��ÿһ���յ����ص���������ΪINF
    for (int row = 0; row < range; row++) {
        for (int col = col_begin; col <= col_end; col++) {
            if ((int)local_mask.at<uchar>(row, col) == 255) {
                energyMMap.at<double>(row, col) = INF;
            }
        }
    }
    //�����ۻ�����
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
    CVMat tmpEnergy; //�ۻ���������
    energyMMap.copyTo(tmpEnergy);
    //���ö�̬�滮�㷨��ȡ��С������
    //��ȡ���һ�е�����ֵ
    vector<pair<int, double>> last_row;
    for (int col = col_begin; col <= col_end; col++) {
        last_row.push_back(make_pair(col, tmpEnergy.at<double>(range - 1, col)));
    }
    //�����һ�е�����ֵ��������
    sort(last_row.begin(), last_row.end(), cmpd);
    int* seam = new int[range];
    //�õ������������һ�е�����
    seam[range - 1] = last_row[0].first;
    //������ǰ���������ߵ����أ�ֻ��Ҫ��������С�ľ�����
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


//��seam����ͼƬ����ȥ
CVMat insertLocalSeam(CVMat src, CVMat& mask, int* seam, SEAMDIRECTION seamDirection, pair<int, int> begin_end, bool shift2end) {
    //�Ƚ��з����ת��������������������������
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
            //�Կ�ȱ����߻����ϱߵ�ϸ�죬��seam��ߵ���������ƽ��һλ
            for (int col = 0; col < seam[local_row]; col++) {
                colorPixel p = src.at<colorPixel>(row, col + 1);
                resimg.at<colorPixel>(row, col) = p;
                mask.at<uchar>(row, col) = mask.at<uchar>(row, col + 1);
            }
        }
        else {
            //�Կ�ȱ���ұ߻����±ߵ�ϸ�죬��seam�ұߵ���������ƽ��һλ
            for (int col = cols - 1; col > seam[local_row]; col--) {
                colorPixel p = src.at<colorPixel>(row, col - 1);
                resimg.at<colorPixel>(row, col) = p;
                mask.at<uchar>(row, col) = mask.at<uchar>(row, col - 1);
            }
        }
        //��seamϸ�츳ֵ��ɫ��
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



//DP�ҵ�������͵�Seam������backward����
int* getLocalSeam(CVMat src, CVMat mask, SEAMDIRECTION seamDirection, pair<int, int> begin_end) {
    //ͳһѰ����ֱ��seam
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
    //Range����ҿ�
    CVMat local_img = displayimg(cv::Range::Range(row_begin, row_end + 1), cv::Range::Range(col_begin, col_end + 1));
    CVMat local_mask = mask(cv::Range::Range(row_begin, row_end + 1), cv::Range::Range(col_begin, col_end + 1));
    //����Sobel���Ӽ���ֲ�ͼ������
    CVMat local_energy = calcEnergy(local_img);
    CVMat local_energy_32f;
    local_energy.convertTo(local_energy_32f, CV_32F);
    //��ȱʧ���ֵ���������Ϊ����
    for (int row = 0; row < range; row++) {
        for (int col = col_begin; col <= col_end; col++) {
            if ((int)local_mask.at<uchar>(row, col) == 255) {
                local_energy_32f.at<float>(row, col) = INF;
            }
        }
    }
    CVMat tmpEnergy;
    local_energy_32f.copyTo(tmpEnergy);
    //���ö�̬�滮�㷨��ȡ��С������
    //���ۼ���������
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
    //��ȡ���һ�е�����ֵ
    vector<pair<int, float>> last_row;
    for (int col = col_begin; col <= col_end; col++) {
        last_row.push_back(make_pair(col, tmpEnergy.at<float>(range - 1, col)));
    }
    //�����һ�е�����ֵ��������
    sort(last_row.begin(), last_row.end(), cmp);
    int* seam = new int[range];
    //�õ������������һ�е�����
    seam[range - 1] = last_row[0].first;
    //������ǰ���������ߵ����أ�ֻ��Ҫ��������С�ľ�����
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



//ȷ��wrapped֮��������
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



//warp back����������������㰴��ƫ�Ƴ���ԭ��ԭͼ
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




//��ȡƫ��������U
vector<vector<CoordinateInt>> getLocalWarpDisplacement(CVMat src, CVMat mask) {

    int rows = src.rows;
    int cols = src.cols;
    //��¼ƫ����������
    vector<vector<CoordinateInt>> displacementMap; //���յ�
    vector<vector<CoordinateInt>> finalDisplacementMap; //��ʱ����
    //��ʼ������
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
            //ͨ����С��������������ȡһ����С������
            //int* seam = getLocalSeam(src, mask, seamdirection, begin_end);
            int* seam = getLocalSeam_improved(src, mask, seamdirection, begin_end);

            //������С�����ߣ�ͬʱ����Ҫλ�Ƶ����ؽ���λ��
            src = insertLocalSeam(src, mask, seam, seamdirection, begin_end, shift2end);

            //����ƫ�ƾ���
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    CoordinateInt tmpdisplacement;
                    if (seamdirection == SEAM_VERTICAL && row >= begin_end.first && row <= begin_end.second) {
                        int local_row = row - begin_end.first;
                        if (col > seam[local_row] && shift2end) {//����
                            tmpdisplacement.col = -1;
                        }
                        else {
                            if (col < seam[local_row] && !shift2end) { //����
                                tmpdisplacement.col = 1;
                            }
                        }
                    }
                    else {
                        if (seamdirection == SEAM_HORIZONTAL && col >= begin_end.first && col <= begin_end.second) {
                            int local_col = col - begin_end.first;
                            if (row > seam[local_col] && shift2end) { //����
                                tmpdisplacement.row = -1;
                            }
                            else {
                                if (row < seam[local_col] && !shift2end) { //����
                                    tmpdisplacement.row = 1;
                                }
                            }
                        }
                    }
                    //׼����¼ƫ����
                    CoordinateInt& finalDisplacement = finalDisplacementMap[row][col];
                    //�����һʱ�̵�����
                    int tmpDisplacement_row = row + tmpdisplacement.row;
                    int tmpDisplacement_col = col + tmpdisplacement.col;
                    //������һʱ�̵�������displacementMap�϶�Ӧ���ȡ��һʱ�̵�ƫ����
                    CoordinateInt displacementOfTarget = displacementMap[tmpDisplacement_row][tmpDisplacement_col];
                    //������һʱ�������ƫ�����������ʼ��
                    int rowOfOrigin = tmpDisplacement_row + displacementOfTarget.row;
                    int colOfOrigin = tmpDisplacement_col + displacementOfTarget.col;
                    //������ʼ������ڵ����������������ڵ�ƫ����
                    finalDisplacement.row = rowOfOrigin - row;
                    finalDisplacement.col = colOfOrigin - col;
                }
            }
            //��finalDisplacementMap���Ƹ�displacementMapȻ�������һ�����ţ�
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