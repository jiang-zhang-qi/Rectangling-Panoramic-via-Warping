#pragma once

#ifndef ComLib_hpp
#define ComLib_hpp
#define GLUT_DISABLE_ATEXIT_HACK
#define INF 1e8
#define PI 3.14159265358979323846
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <algorithm>
#include "lsd.h"
#include <time.h>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <cmath>
#include <GL/glut.h>

typedef cv::Mat CVMat;
typedef cv::Vec3b colorPixel;

typedef Eigen::SparseMatrix<double> SparseMatrixD; //����չ�����������ݴ���
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseMatrixDRow; //����չ�����������ݴ���
typedef Eigen::VectorXd VectorXd; //��̬����
typedef Eigen::MatrixXd MatrixXd; //��̬����
typedef Eigen::Vector2d Vector2d; //2ά����
typedef Eigen::Vector2i Vector2i;
typedef Eigen::MatrixXi MatrixXi;
typedef Eigen::Matrix2d Matrix2d; // 2x2matrix of double
typedef Eigen::SimplicialCholesky<SparseMatrixD> CSolve; //Cholesky���ϡ����������ֵ

using namespace std;

//ͼ�δ�С�Լ�mesh����
struct Config {
    int rows;
    int cols;
    int meshNumRow; //����ÿ���ж��ٵ�
    int meshNumCol;
    int meshQuadRow; //һ�����ж��ٸ�С����
    int meshQuadCol;
    double rowPerMesh;
    double colPerMesh;
    Config(int rows, int cols, int meshNumRow, int meshNumCol) {
        this->rows = rows;
        this->cols = cols;
        this->meshNumCol = meshNumCol;
        this->meshNumRow = meshNumRow;
        this->meshQuadCol = meshNumCol - 1;
        this->meshQuadRow = meshNumRow - 1;
        //��������Ϊʲô��Ҫ-1����Ϊ��֮����ȷ��������ʱ�����һ�����ǳ��Ϸ�ĸ�ģ������-1�ᵼ�����һ����rows��cols
        //�������Ǵ�0��ʼ�ģ����Իᳬ��rows-1��cols-1������Χ�ˣ���
        this->rowPerMesh = double(rows - 1) / (meshNumRow - 1);
        this->colPerMesh = double(cols - 1) / (meshNumCol - 1);
    }
};

//�������꣬������¼���ص�ƫ����
struct CoordinateInt {
    int row;
    int col;

    bool operator==(const CoordinateInt& cd) const {
        return (row == cd.row && col == cd.col);
    }
    bool operator<(const CoordinateInt& cd) const {
        if (row < cd.row) {
            return true;
        }
        if (row > cd.row) {
            return false;
        }
        return col < cd.col;
    }
    CoordinateInt() {
        row = 0;
        col = 0;
    }
    CoordinateInt(int setRow, int setCol) {
        row = setRow;
        col = setCol;
    }
};

//��������
struct CoordinateDouble {
    double row;
    double col;

    bool operator==(const CoordinateDouble& rhs) const {
        return (row == rhs.row && col == rhs.col);
    }
    bool operator<(const CoordinateDouble& rhs) const {
        // this operator is used to determine equality, so it must use both x and y
        if (row < rhs.row) {
            return true;
        }
        if (row > rhs.row) {
            return false;
        }
        return col < rhs.col;
    }

    CoordinateDouble operator+(const CoordinateDouble& b)
    {
        CoordinateDouble temp;
        temp.row = row + b.row;
        temp.col = col + b.col;
        return temp;
    }
    CoordinateDouble operator-(const CoordinateDouble& b)
    {
        CoordinateDouble temp;
        temp.row = row - b.row;
        temp.col = col - b.col;
        return temp;
    }

    friend ostream& operator<<(ostream& stream, const CoordinateDouble& p) {
        stream << "(" << p.col << "," << p.row << ")";
        return stream;
    }
    CoordinateDouble() {
        row = 0;
        col = 0;
    }
    CoordinateDouble(double setRow, double setCol) {
        row = setRow;
        col = setCol;
    }
};

//����ȷ��һ���߶�
struct LineD {
    double row1, col1;
    double row2, col2;
    LineD(double row1, double col1, double row2, double col2) {
        this->row1 = row1;
        this->row2 = row2;
        this->col1 = col1;
        this->col2 = col2;
    }
    LineD() {
        row1 = 0; col1 = 0;
        row2 = 0; col2 = 0;
    }
    LineD(CoordinateDouble p1, CoordinateDouble p2) {
        row1 = p1.row; col1 = p1.col;
        row2 = p2.row; col2 = p2.col;
    }
};

CVMat MaskContour(const CVMat src);
vector<vector<CoordinateDouble>> vector2mesh(VectorXd x, Config config);
SparseMatrixDRow VStack(SparseMatrixD origin, SparseMatrixDRow diag);
SparseMatrixDRow VStack(SparseMatrixDRow origin, SparseMatrixDRow diag);
MatrixXd VStack(MatrixXd mat1, MatrixXd mat2);
MatrixXd HStack(MatrixXd mat1, MatrixXd mat2);
void DrawLine(CVMat& img, CoordinateDouble coordstart, CoordinateDouble coordend);
void DrawLine(CVMat& img, LineD line);
void enlargeMesh(vector<vector<CoordinateDouble>>& mesh, double enlarge_x, double enlarge_y, Config config);
void DrawMesh(const CVMat src, vector<vector<CoordinateDouble>> mesh, Config config);
void computeScaling(double& sx_avg, double& sy_avg, const vector<vector<CoordinateDouble>> mesh, const vector<vector<CoordinateDouble>> outputmesh, const Config config);
CVMat myimReadfun(string path, double& sampleScaled, double scale, bool& needScaled, bool flag, int& viewS, int numviewS);


#endif // !ComLib_hpp
#pragma once
