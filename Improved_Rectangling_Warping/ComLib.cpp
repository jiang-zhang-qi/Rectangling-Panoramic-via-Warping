#include "ComLib.h"

void fillHole(const CVMat srcBw, CVMat& dstBw) {
    cv::Size m_Size = srcBw.size();
    CVMat Temp = CVMat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());
    srcBw.copyTo(Temp(cv::Range::Range(1, m_Size.height + 1), cv::Range::Range(1, m_Size.width + 1)));
    cv::floodFill(Temp, cv::Point(0, 0), cv::Scalar(255));

    CVMat cutImg;
    Temp(cv::Range::Range(1, m_Size.height + 1), cv::Range::Range(1, m_Size.width + 1)).copyTo(cutImg);

    //cv::imshow("cutImg", ~cutImg);
    dstBw = srcBw | (~cutImg);
}


//input:不规则照片，output：不规则照片的边界
CVMat MaskContour(const CVMat src) {
    CVMat bw;
    CVMat src_gray;
    //将三色图转为灰度图
    src.copyTo(src_gray);
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
    //cv::imshow("Gray", src);
    //这里为什么不设置阈值为255呢？因为为了抗锯齿！！！
    //因为图像边缘有很多的锯齿，非平滑的，不是纯白的就直接设置成图像颜色，导致锯齿很明显！！
    uchar thr = 252;
    //创建mask图像
    CVMat mask = CVMat::zeros(src_gray.size(), CV_8UC1);
    for (int row = 0; row < src_gray.rows; row++) {
        for (int col = 0; col < src_gray.cols; col++) {
            if (src_gray.at<uchar>(row, col) < thr) {
                mask.at<uchar>(row, col) = 255;
            }
        }
    }
    //通过漫灌，对空缺部分进行填充，空缺部分为黑
    fillHole(mask, bw);
    //将空缺部分置白
    bw = ~bw;
    //膨胀腐蚀部分可以处理边缘琐碎细节，抗锯齿
    CVMat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    CVMat dilate_out;//膨胀
    cv::dilate(bw, dilate_out, element);
    cv::dilate(dilate_out, dilate_out, element);
    cv::dilate(dilate_out, dilate_out, element);

    CVMat erode_out;//腐蚀
    erode(dilate_out, erode_out, element);

    return erode_out;
}

//将线性的向量转化为矩阵，这里就是转化为网格点
//1xm*n*2 -> mxn的mesh
vector<vector<CoordinateDouble>> vector2mesh(VectorXd x, Config config) {
    int numMeshRow = config.meshNumRow;
    int numMeshCol = config.meshNumCol;
    vector<vector<CoordinateDouble>> mesh;
    for (int row = 0; row < numMeshRow; row++) {
        vector<CoordinateDouble> meshRow;
        for (int col = 0; col < numMeshCol; col++) {
            int xid = (row * numMeshCol + col) * 2;
            CoordinateDouble coord;
            coord.row = x(xid + 1);
            coord.col = x(xid);
            meshRow.push_back(coord);
        }
        mesh.push_back(meshRow);
    }
    return mesh;
}

//VSsack用于将两个矩阵上下拼接起来
//Vertical stack
SparseMatrixDRow VStack(SparseMatrixD origin, SparseMatrixDRow diag) {
    SparseMatrixDRow res(origin.rows() + diag.rows(), origin.cols());
    res.topRows(origin.rows()) = origin;
    res.bottomRows(diag.rows()) = diag;
    return res;
}
SparseMatrixDRow VStack(SparseMatrixDRow origin, SparseMatrixDRow diag) {
    SparseMatrixDRow res(origin.rows() + diag.rows(), origin.cols());
    res.topRows(origin.rows()) = origin;
    res.bottomRows(diag.rows()) = diag;
    return res;
}
MatrixXd VStack(MatrixXd mat1, MatrixXd mat2) {
    MatrixXd res(mat1.rows() + mat2.rows(), mat1.cols());
    res.topRows(mat1.rows()) = mat1;
    res.bottomRows(mat2.rows()) = mat2;
    return res;
}
//Horizontal Stack
MatrixXd Hstack(MatrixXd mat1, MatrixXd mat2) {
    MatrixXd res(mat1.rows(), mat1.cols() + mat2.cols());
    res.leftCols(mat1.cols()) = mat1;
    res.rightCols(mat2.cols()) = mat2;
    return res;
}

void DrawLine(CVMat& img, CoordinateDouble coordstart, CoordinateDouble coordend) {
    cv::Point start((int)coordstart.col, (int)coordstart.row);
    cv::Point end((int)coordend.col, (int)coordend.row);
    int thickness = 1;
    int lineType = cv::LINE_AA;
    cv::line(img, start, end, cv::Scalar(0, 255, 0), thickness, lineType);
}

void DrawLine(CVMat& img, LineD line) {
    cv::Point start((int)line.col1, (int)line.row1);
    cv::Point end((int)line.col2, (int)line.row2);
    int thickness = 1;
    int lineType = cv::LINE_AA;
    cv::line(img, start, end, cv::Scalar(0, 255, 0), thickness, lineType);
}

//绘制网格线
void DrawMesh(const CVMat src, vector<vector<CoordinateDouble>> mesh, Config config) {
    CVMat src_copy;
    src.copyTo(src_copy);
    int meshLineRow = config.meshNumRow;
    int meshLineCol = config.meshNumCol;

    for (int row = 0; row < meshLineRow; row++) {
        for (int col = 0; col < meshLineCol; col++) {
            CoordinateDouble now = mesh[row][col];
            if (row == meshLineRow - 1 && col < meshLineCol - 1) { //mesh的底边
                CoordinateDouble right = mesh[row][col + 1];
                DrawLine(src_copy, now, right);
            }
            else if (row < meshLineRow - 1 && col == meshLineCol - 1) { //mesh的右侧边
                CoordinateDouble down = mesh[row + 1][col];
                DrawLine(src_copy, now, down);
            }
            else if (row < meshLineRow - 1 && col < meshLineCol - 1) { //除了上述两种特殊点，其余点只要往右和往下画线就可以了！
                CoordinateDouble right = mesh[row][col + 1];
                DrawLine(src_copy, now, right);
                CoordinateDouble down = mesh[row + 1][col];
                DrawLine(src_copy, now, down);
            }
        }
    }
    cv::namedWindow("Mesh", cv::WINDOW_NORMAL);
    cv::imshow("Mesh", src_copy);
    cv::waitKey(0);
}


//根据x y的因子调整mesh
void enlargeMesh(vector<vector<CoordinateDouble>>& mesh, double enlarge_x, double enlarge_y, Config config) {
    int numMeshRow = config.meshNumRow;
    int numMeshCol = config.meshNumCol;
    for (int row = 0; row < numMeshRow; row++) {
        for (int col = 0; col < numMeshCol; col++) {
            CoordinateDouble& coord = mesh[row][col];

            coord.row = coord.row * enlarge_y;
            coord.col = coord.col * enlarge_x;
        }
    }
}

//计算论文中最后提到的缩放因子
void computeScaling(double& sx_avg, double& sy_avg, const vector<vector<CoordinateDouble>> mesh, const vector<vector<CoordinateDouble>> outputmesh, const Config config)
{
    //row为y轴，col为x轴
    int numQuadRow = config.meshQuadRow;
    int numQuadCol = config.meshQuadRow;
    int sx = 0, sy = 0;
    for (int row = 0; row < numQuadRow; row++)
    {
        for (int col = 0; col < numQuadCol; col++)
        {
            CoordinateDouble p0 = mesh[row][col];//左上
            CoordinateDouble p1 = mesh[row][col + 1];//右上
            CoordinateDouble p2 = mesh[row + 1][col];//左下
            CoordinateDouble p3 = mesh[row + 1][col + 1];//右下

            CoordinateDouble p0_out = outputmesh[row][col];//左上
            CoordinateDouble p1_out = outputmesh[row][col + 1];//右上
            CoordinateDouble p2_out = outputmesh[row + 1][col];//左下
            CoordinateDouble p3_out = outputmesh[row + 1][col + 1];//右下

            //calculate y scaled factor
            CVMat A = (cv::Mat_<double>(1, 4) << p0.row, p1.row, p2.row, p3.row);
            CVMat B = (cv::Mat_<double>(1, 4) << p0_out.row, p1_out.row, p2_out.row, p3_out.row);
            double max_temp, min_temp;
            double max_temp_out, min_temp_out;
            cv::minMaxIdx(A, &max_temp, &min_temp);
            cv::minMaxIdx(B, &max_temp_out, &min_temp_out);
            sy += (max_temp_out - min_temp_out) / (max_temp - min_temp);
            //calculate x scaled factor
            CVMat C = (cv::Mat_<double>(1, 4) << p0.col, p1.col, p2.col, p3.col);
            CVMat D = (cv::Mat_<double>(1, 4) << p0_out.col, p1_out.col, p2_out.col, p3_out.col);
            max_temp = 0; min_temp = 0; max_temp_out = 0; min_temp_out = 0;
            cv::minMaxIdx(C, &max_temp, &min_temp);
            cv::minMaxIdx(D, &max_temp_out, &min_temp_out);
            sx += (max_temp_out - min_temp_out) / (max_temp - min_temp);
        }
    }
    sx_avg = double(sx) / (numQuadRow * numQuadCol);
    sy_avg = double(sy) / (numQuadRow * numQuadCol);
}

//自己定义的照片读取函数，同时定义了样本下采样的比率，和最终用openGL展现图片时的缩放比率
CVMat myimReadfun(string path, double& sampleScaled, double scale, bool& needScaled, bool flag, int& viewS, int numviewS) {
    CVMat img = cv::imread(path);
    sampleScaled = scale;
    needScaled = flag;
    viewS = numviewS;
    return img;
}