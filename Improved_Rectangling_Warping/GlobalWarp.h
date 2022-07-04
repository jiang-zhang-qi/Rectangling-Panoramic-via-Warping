#pragma once
#ifndef GlobalWarp_hpp
#define GlobalWarp_hpp

#include "ComLib.h"

#define clamp(x,a,b)    (  ((a)<(b))				\
? ((x)<(a))?(a):(((x)>(b))?(b):(x))	\
: ((x)<(b))?(b):(((x)>(a))?(a):(x))	\
)

struct Line_rotate {
	Vector2d pstart = Vector2d::Zero();
	Vector2d pend = Vector2d::Zero();
	double angle = 0;
	Line_rotate(Vector2d pstart, Vector2d pend, double angle) {
		this->pstart = pstart;
		this->pend = pend;
		this->angle = angle;
	}
};

struct BilinearWeights {
	double s;
	double t;
};

struct myBilinearWeights {
	double u;
	double v;
	myBilinearWeights(double u, double v) {
		this->u = u;
		this->v = v;
	}
};



myBilinearWeights getMyBilinearWeights(CoordinateDouble point, CoordinateInt upperLeftIndices, vector<vector<CoordinateDouble>> mesh);
SparseMatrixDRow getShapeMat(vector<vector<CoordinateDouble>> mesh, Config config);
SparseMatrixDRow getVertex2ShapeMat(vector<vector<CoordinateDouble>> mesh, Config config);
pair<SparseMatrixDRow, VectorXd> getBoundaryMat(const CVMat src, vector<vector<CoordinateDouble>> mesh, Config config);
VectorXd getVertice(int row, int col, vector<vector<CoordinateDouble>> mesh);
vector<vector<vector<LineD>>> initLineSeg(CVMat src, CVMat mask, Config config, vector<LineD>& lineSeg_flatten, vector<vector<CoordinateDouble>> mesh, vector<pair<int, double>>& id_theta, vector<double>& rotate_theta);
SparseMatrixDRow getLineMat(CVMat src, CVMat mask, vector<vector<CoordinateDouble>> mesh, vector<double> rotate_theta, vector<vector<vector<LineD>>> lineSeg, vector<pair<MatrixXd, MatrixXd>>& BilinearVec, Config config, int& lineNum, vector<bool>& bad);
//double cross(CoordinateDouble a, CoordinateDouble b);
//InvBilinearWeights getBilinearWeights(CoordinateDouble point, CoordinateDouble upperLeftIndices, const vector<vector<CoordinateDouble>>& mesh);
//MatrixXd IBilinearWeights2Matrix(InvBilinearWeights w);
bool isInQuad(CoordinateDouble point, CoordinateDouble topLeft, CoordinateDouble topRight, CoordinateDouble bottomLeft, CoordinateDouble bottomRight);
CVMat fillMissingPixel(CVMat& img, const CVMat mask);


#endif // !GlobalWarp_hpp
