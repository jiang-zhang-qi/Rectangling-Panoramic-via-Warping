#include "GlobalWarp.h"

//Boundary Constraints
pair<SparseMatrixDRow, VectorXd> getBoundaryMat(CVMat src, vector<vector<CoordinateDouble>> mesh, Config config) {
	//Vq = [x0, y0, x1, y2 ...]
	int rows = config.rows;
	int cols = config.cols;
	int meshRowNum = config.meshNumRow; //20
	int meshColNum = config.meshNumCol; //20
	int vertexNum = meshRowNum * meshColNum; //400
	//������������(x, y)����Ҫ��2��
	VectorXd dvec = VectorXd::Zero(vertexNum * 2);
	VectorXd value = VectorXd::Zero(vertexNum * 2);

	for (int i = 0; i < vertexNum * 2; i += meshColNum * 2) { //������ߵĵ�
		//��������x=0
		dvec(i) = 1;
		value(i) = 0;
	}
	for (int i = meshColNum * 2 - 2; i < vertexNum * 2; i += meshColNum * 2) {//�����ұߵĵ�
		//��������x=cols - 1
		dvec(i) = 1;
		value(i) = cols - 1;  //w
	}
	for (int i = 1; i < meshColNum * 2; i += 2) {//������
		//��������y=0
		dvec(i) = 1;
		value(i) = 0;
	}
	for (int i = vertexNum * 2 - meshColNum * 2 + 1; i < vertexNum * 2; i += 2) {//������
		//��������y = rows - 1��
		dvec(i) = 1;
		value(i) = rows - 1; //h
	}
	SparseMatrixDRow diag(dvec.size(), dvec.size()); //800x800�ĶԽǾ���
	for (int i = 0; i < dvec.size(); i++) {
		diag.insert(i, i) = dvec(i);
	}
	diag.makeCompressed();
	return make_pair(diag, value);

}

//��ȡShape Energy��ϵ���������
/*
		Aq(AqT* Aq) - 1AqT - I
*/
SparseMatrixDRow getShapeMat(vector<vector<CoordinateDouble>> mesh, Config config) {
	int quadRowNum = config.meshQuadRow;
	int quadColNum = config.meshQuadCol;//��������
	//��Ϊÿ��������4���㣨4�����꣩��8������Aq
	SparseMatrixDRow shapeEnergy(8 * quadRowNum * quadColNum, 8 * quadRowNum * quadColNum);
	for (int row = 0; row < quadRowNum; row++) {
		for (int col = 0; col < quadColNum; col++) {
			//��ȡ��ǰ������ĸ���
			CoordinateDouble p0 = mesh[row][col]; //����
			CoordinateDouble p1 = mesh[row][col + 1]; //����
			CoordinateDouble p2 = mesh[row + 1][col]; //����
			CoordinateDouble p3 = mesh[row + 1][col + 1]; //����
			MatrixXd Aq(8, 4);
			Aq << p0.col, -p0.row, 1, 0,
				p0.row, p0.col, 0, 1,
				p1.col, -p1.row, 1, 0,
				p1.row, p1.col, 0, 1,
				p2.col, -p2.row, 1, 0,
				p2.row, p2.col, 0, 1,
				p3.col, -p3.row, 1, 0,
				p3.row, p3.col, 0, 1;
			MatrixXd Aq_trans = Aq.transpose();
			MatrixXd Aq_trans_mul_Aq_reverse = (Aq_trans * Aq).inverse();
			MatrixXd I = MatrixXd::Identity(8, 8);
			MatrixXd coeff = (Aq * (Aq_trans_mul_Aq_reverse)*Aq_trans - I);
			//�ѵ�ǰ�����energy 8*8���뵽shapeEnergy�������,ֻ���˶ԽǾ���
			/*
			[ [           ]                                ]
			| |           |                                |
			| |           |                                |
			| [           ]                                |
			|             [           ]                    |
			|             |           |                    |
			|             |           |                    |
			|             [           ]                    |
		   ...							                  ...
			|                                 [          ] |
			|                                 |          | |
			|                                 |          | |
			[                                 [          ] ]
			*/
			int left_top_x = (row * quadColNum + col) * 8;
			for (int i = 0; i < 8; i++) {
				for (int j = 0; j < 8; j++) {
					shapeEnergy.insert(left_top_x + i, left_top_x + j) = coeff(i, j);
				}
			}
		}
	}
	shapeEnergy.makeCompressed();
	return shapeEnergy;
}


//��quad�ĵ����껯��8x1�ľ���
VectorXd getVertice(int row, int col, vector<vector<CoordinateDouble>> mesh) {
	VectorXd Vq = VectorXd::Zero(8);
	CoordinateDouble p0 = mesh[row][col];
	CoordinateDouble p1 = mesh[row][col + 1];
	CoordinateDouble p2 = mesh[row + 1][col];
	CoordinateDouble p3 = mesh[row + 1][col + 1];
	Vq << p0.col, p0.row, p1.col, p1.row, p2.col, p2.row, p3.col, p3.row;
	return Vq;
}

//�Ժ����б任�õ�,��¼����ͼ��quad����ֲ�
SparseMatrixDRow getVertex2ShapeMat(vector<vector<CoordinateDouble>> mesh, Config config) {
	int meshRowNum = config.meshNumRow;
	int meshColNum = config.meshNumCol;
	int quadRowNum = config.meshQuadRow;
	int quadColNum = config.meshQuadCol;
	SparseMatrixDRow Q(8 * quadRowNum * quadColNum, 2 * meshRowNum * meshColNum);
	for (int row = 0; row < quadRowNum; row++) {
		for (int col = 0; col < quadColNum; col++) {
			int quadId = 8 * (row * quadColNum + col); //��ǰ������
			int topLeftVerId = 2 * (row * meshColNum + col); //��ǰ�������Ͻ�x����
			//����
			Q.insert(quadId, topLeftVerId) = 1;
			Q.insert(quadId + 1, topLeftVerId + 1) = 1;
			//����
			Q.insert(quadId + 2, topLeftVerId + 2) = 1;
			Q.insert(quadId + 3, topLeftVerId + 3) = 1;
			//����
			Q.insert(quadId + 4, topLeftVerId + 2 * meshColNum) = 1;
			Q.insert(quadId + 5, topLeftVerId + 2 * meshColNum + 1) = 1;
			//����
			Q.insert(quadId + 6, topLeftVerId + 2 * meshColNum + 2) = 1;
			Q.insert(quadId + 7, topLeftVerId + 2 * meshColNum + 3) = 1;
		}
	}
	//Q��¼ÿ��Quad��Ӧ��Vq
	Q.makeCompressed();
	return Q;
}

//�ж�a�Ƿ���x0��x1֮��
bool between(double a, double x0, double x1) {
	double temp1 = a - x0;
	double temp2 = a - x1;

	if ((temp1 < 1e-8 && temp2 > -1e-8) || (temp2 < 1e-6 && temp1 > -1e-8)) {
		return true;
	}
	else {
		return false;
	}
}

//��������߶��Ƿ��н���
Vector2d detectIntersect(Matrix2d line1, Matrix2d line2, bool& isintersection) {
	double line_x = 0, line_y = 0;
	double p1_x = line1(0, 1), p1_y = line1(0, 0), p2_x = line1(1, 1), p2_y = line1(1, 0);
	double p3_x = line2(0, 1), p3_y = line2(0, 0), p4_x = line2(1, 1), p4_y = line2(1, 0);
	if ((fabs(p1_x - p2_x) < 1e-6) && (fabs(p3_x - p4_x) < 1e-6)) { //˵������x�ᴹֱ��ƽ��
		isintersection = false;
	}
	else if (fabs(p1_x - p2_x) < 1e-6) { //line1��x�ᴹֱ
		if (between(p1_x, p3_x, p4_x)) {
			double k = (p4_y - p3_y) / (p4_x - p3_x);
			line_x = p1_x;
			line_y = k * (line_x - p3_x) + p3_y;

			if (between(line_y, p1_y, p2_y)) {
				isintersection = true;
			}
			else {
				isintersection = false;
			}
		}
		else {
			isintersection = false;
		}
	}
	else if (fabs(p3_x - p4_x) < 1e-6) { //line2��x�ᴹֱ
		if (between(p3_x, p1_x, p2_x)) {
			double k = (p2_y - p1_y) / (p2_x - p1_x);
			line_x = p3_x;
			line_y = k * (line_x - p1_x) + p1_y;
			if (between(line_y, p3_y, p4_y)) {
				isintersection = true;
			}
			else {
				isintersection = false;
			}
		}
		else {
			isintersection = false;
		}
	}
	else {
		double k1 = (p2_y - p1_y) / (p2_x - p1_x);
		double k2 = (p4_y - p3_y) / (p4_x - p3_x);
		if (fabs(k1 - k2) < 1e-6) {
			isintersection = false;
		}
		else {
			line_x = ((p3_y - p1_y) - (k2 * p3_x - k1 * p1_x)) / (k1 - k2);
			line_y = k1 * (line_x - p1_x) + p1_y;
		}

		if (between(line_x, p1_x, p2_x) && between(line_x, p3_x, p4_x)) {
			isintersection = true;
		}
		else {
			isintersection = false;
		}
	}
	Vector2d p;
	p << line_y, line_x;
	return p;
}

//����ע��Ե�ߵļ��
void reviseMask4Lines(CVMat& mask) {
	int rows = mask.rows;
	int cols = mask.cols;
	//����Ե�ð�
	for (int row = 0; row < rows; row++) {
		mask.at<uchar>(row, 0) = 255;
		mask.at<uchar>(row, cols - 1) = 255;
	}
	for (int col = 0; col < cols; col++) {
		mask.at<uchar>(0, col) = 255;
		mask.at<uchar>(rows - 1, col) = 255;
	}
	//�൱��һ��8x8��kernel
	CVMat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 8));
	//��mask��������,���ڰ�ɫԪ����˵�����͵�
	cv::dilate(mask, mask, element);
	cv::dilate(mask, mask, element);
}

//�жϵ��Ƿ���Quad��
bool isInQuad(CoordinateDouble point, CoordinateDouble topLeft, CoordinateDouble topRight, CoordinateDouble bottomLeft, CoordinateDouble bottomRight) {

	//colΪx�ᣬrowΪy��
	//�Ƿ����������ߵ��ҷ�
	if (topLeft.col == bottomLeft.col) { //��������ƽ��
		if (point.col < topLeft.col) { //
			return false;
		}
	}
	else {
		double leftSlope = (topLeft.row - bottomLeft.row) / (topLeft.col - bottomLeft.col);
		double leftIntersect = topLeft.row - leftSlope * topLeft.col;
		double yOnLineX = (point.row - leftIntersect) / leftSlope;
		if (point.col < yOnLineX) {
			return false;
		}
	}
	//�Ƿ��������Ҳ����
	if (topRight.col == bottomRight.col) {
		if (point.col > topRight.col) {
			return false;
		}
	}
	else {
		double rightSlope = (topRight.row - bottomRight.row) / (topRight.col - bottomRight.col);
		double rightIntersect = topRight.row - topRight.col * rightSlope;
		double yOnLineX = (point.row - rightIntersect) / rightSlope;
		if (point.col > yOnLineX) {
			return false;
		}
	}
	//�Ƿ��������ϲ���·�
	if (topLeft.row == topRight.row) {
		if (point.row < topLeft.row) {
			return false;
		}
	}
	else {
		double topSlope = (topRight.row - topLeft.row) / (topRight.col - topLeft.col);
		double topIntersect = topRight.row - topSlope * topRight.col;
		double xOnlineY = topSlope * point.col + topIntersect;
		if (point.row < xOnlineY) {
			return false;
		}
	}
	//�Ƿ��������²���Ϸ�
	if (bottomLeft.row == bottomRight.row) {
		if (point.row > bottomLeft.row) {
			return false;
		}
	}
	else {
		double bottomSlope = (bottomRight.row - bottomLeft.row) / (bottomRight.col - bottomLeft.col);
		double bottomIntersect = bottomRight.row - bottomSlope * bottomRight.col;
		double xOnlineY = bottomSlope * point.col + bottomIntersect;
		if (point.row > xOnlineY) {
			return false;
		}
	}
	return true;
}

//�ж��߶������Ƿ�λ�����ֵ���Ч����
bool lineInMask(CVMat mask, LineD line) {
	int row1 = round(line.row1), row2 = round(line.row2), col1 = round(line.col1), col2 = round(line.col2);
	if ((mask.at<uchar>(row1, col1) != 0) && (mask.at<uchar>(row2, col2) != 0)) return false;
	if ((col1 == mask.cols - 1 && col2 == mask.cols - 1) || (col1 == 0 && col2 == 0)) {
		return false;
	}
	if ((row1 == mask.rows - 1 && row2 == mask.rows - 1) || (row1 == 0 && row2 == 0)) {
		return false;
	}
	//��һ���˵��ڱ߽�
	if (row1 == 0 || row1 == mask.rows - 1 || col1 == mask.cols - 1 || col1 == 0) {
		try {
			if (mask.at<uchar>(row2 + 1, col2) == 255 || mask.at<uchar>(row2 - 1, col2) == 255 || mask.at<uchar>(row2, col2 + 1) == 255 || mask.at<uchar>(row2, col2 - 1) == 255) {
				return false;
			}
		}
		catch (std::exception) {}
		return true;
	}
	if (row2 == 0 || row2 == mask.rows - 1 || col2 == 0 || col2 == mask.cols - 1) {
		try {
			if (mask.at<uchar>(row1 + 1, col1) == 255 || mask.at<uchar>(row1 - 1, col1) == 255 || mask.at<uchar>(row1, col1 + 1) == 255 || mask.at<uchar>(row1, col1 - 1) == 255) {
				return false;
			}
		}
		catch (std::exception) {}
		return true;
	}
	//һ�����
	try {
		if (mask.at<uchar>(row1 + 1, col1) == 255 || mask.at<uchar>(row1 - 1, col1) == 255 || mask.at<uchar>(row1, col1 + 1) == 255 || mask.at<uchar>(row1, col1 - 1) == 255) {
			return false;
		}
		else {
			if (mask.at<uchar>(row2 + 1, col2) == 255 || mask.at<uchar>(row2 - 1, col2) == 255 || mask.at<uchar>(row2, col2 + 1) == 255 || mask.at<uchar>(row2, col2 - 1) == 255) {
				return false;
			}
			else {
				return true;
			}
		}
	}
	catch (std::exception) {
		throw "line judge fall!";
	}
}

//ͨ��lsd.h������߶Σ��������߶�����
vector<LineD> lsdDetect(CVMat src, CVMat mask) {
	int rows = src.rows;
	int cols = src.cols;
	CVMat gray_img;
	cv::cvtColor(src, gray_img, cv::COLOR_BGR2GRAY);
	double* image = new double[gray_img.cols * gray_img.rows];
	//���Ҷ�ͼ��ƽ��һά����flatten
	for (int row = 0; row < gray_img.rows; row++) {
		for (int col = 0; col < gray_img.cols; col++) {
			image[row * gray_img.cols + col] = gray_img.at<uchar>(row, col);
		}
	}
	vector<LineD> lines;
	double* out;
	int numLines;
	//ͨ��lsd�����߶μ�⣬�õ��߶���numLines�Լ�ÿ���߶ζ�Ӧ�Ķ�������
	out = lsd(&numLines, image, gray_img.cols, gray_img.rows);
	for (int i = 0; i < numLines; i++) {
		//x1, y1, x2, y2, width, p, -log_nfa
		LineD line(out[i * 7 + 1], out[i * 7], out[i * 7 + 3], out[i * 7 + 2]);
		if (lineInMask(mask, line)) {
			lines.push_back(line);
		}
	}
	return lines;
}

//�ж��߶κ�����(�ϡ��¡�����)ֱ�ߣ������߶Σ��Ƿ��н���
bool doesSegmentIntersectLine(LineD lineSegment, double slope, double intersect, bool vertical, CoordinateDouble& intersectPoint) {
	double lineSegmentSlope = INF;
	//ֻҪ������ֱ����ôб�ʾ��Ǵ��ڵ�
	if (lineSegment.col1 != lineSegment.col2) {
		lineSegmentSlope = (lineSegment.row2 - lineSegment.row1) / (lineSegment.col2 - lineSegment.col1);
	}
	//�߶εĽؾ�
	double lineSegmentIntersect = lineSegment.row1 - lineSegmentSlope * lineSegment.col1;

	//����߶�б�ʺ�quadб����ȣ�������ƽ��
	if (lineSegmentSlope == slope) {
		if (lineSegmentIntersect == intersect) { //�������غ�
			intersectPoint.col = lineSegment.col1;
			intersectPoint.row = lineSegment.row1;
			return true;
		}
		else {
			return false;
		}
	}
	//intersectX �� intersectY �ǽ����x y����
	double intersectX = (intersect - lineSegmentIntersect) / (lineSegmentSlope - slope);
	double intersectY = lineSegmentSlope * intersectX + lineSegmentIntersect;
	if (vertical) { //���Ҵ�ֱ����
		if ((intersectY <= lineSegment.row1 && intersectY >= lineSegment.row2) || (intersectY <= lineSegment.row2 && intersectY >= lineSegment.row1)) {
			intersectPoint.col = intersectX;
			intersectPoint.row = intersectY;
			return true;
		}
		else {
			return false;
		}
	}
	else { //�ϡ��µ�ˮƽ��
		if ((intersectX <= lineSegment.col1 && intersectX >= lineSegment.col2) || (intersectX <= lineSegment.col2 && intersectY >= lineSegment.col1)) {
			intersectPoint.col = intersectX;
			intersectPoint.row = intersectY;
			return true;
		}
		else {
			return false;
		}
	}
}

//�����߶κ�����Ľ��㣨0 or 1 or 2���������ؽ�������
vector<CoordinateDouble> intersectionWithQuad(LineD lineSegment, CoordinateDouble topLeft, CoordinateDouble topRight, CoordinateDouble bottomLeft, CoordinateDouble bottomRight) {
	//��¼������Ϣ
	vector<CoordinateDouble> intersections;
	//���������ߵĽ���
	double leftSlope = INF;
	if (topLeft.col != bottomLeft.col) {
		leftSlope = (topLeft.row - bottomLeft.row) / (topLeft.col - bottomLeft.col);
	}
	double leftIntersect = topLeft.row - leftSlope * topLeft.col;
	CoordinateDouble leftIntersectPoint;
	if (doesSegmentIntersectLine(lineSegment, leftSlope, leftIntersect, true, leftIntersectPoint)) {
		if (leftIntersectPoint.row >= topLeft.row && leftIntersectPoint.row <= bottomLeft.row) {
			intersections.push_back(leftIntersectPoint);
		}
	}

	//���������߽���
	double rightSlope = INF;
	if (topRight.col != bottomRight.col) {
		rightSlope = (topRight.row - bottomRight.row) / (topRight.col - bottomRight.col);
	}
	double rightIntersect = topRight.row - rightSlope * topRight.col;
	CoordinateDouble rightIntersectPoint;
	if (doesSegmentIntersectLine(lineSegment, rightSlope, rightIntersect, true, rightIntersectPoint)) {
		if (rightIntersectPoint.row >= topRight.row && rightIntersectPoint.row <= bottomRight.row) {
			intersections.push_back(rightIntersectPoint);
		}
	}

	//���������ߵĽ���
	double topSlope = INF;
	if (topLeft.col != topRight.col) {
		topSlope = (topRight.row - topLeft.row) / (topRight.col - topLeft.col);
	}
	double topIntersect = topLeft.row - topSlope * topLeft.col;
	CoordinateDouble topIntersectPoint;
	if (doesSegmentIntersectLine(lineSegment, topSlope, topIntersect, false, topIntersectPoint)) {
		if (topIntersectPoint.col >= topLeft.col && topIntersectPoint.col <= topRight.col) {
			intersections.push_back(topIntersectPoint);
		}
	}

	//���������ߵĽ���
	double bottomSlope = INF;
	if (bottomLeft.col != bottomRight.col) {
		bottomSlope = (bottomRight.row - bottomLeft.row) / (bottomRight.col - bottomLeft.col);
	}
	double bottomIntersect = bottomLeft.row - bottomSlope * bottomLeft.col;
	CoordinateDouble bottomIntersectPoint;
	if (doesSegmentIntersectLine(lineSegment, bottomSlope, bottomIntersect, false, bottomIntersectPoint)) {
		if (bottomIntersectPoint.col >= bottomLeft.col && bottomIntersectPoint.col <= bottomRight.col) {
			intersections.push_back(bottomIntersectPoint);
		}
	}

	return intersections;
}

//���߷ֶΣ�����ÿ��������
vector<vector<vector<LineD>>> segmentLineInQuad(CVMat src, vector<LineD> lines, vector<vector<CoordinateDouble>> mesh, Config config) {
	int quadRowNum = config.meshQuadRow;
	int quadColNum = config.meshQuadCol;

	vector<vector<vector<LineD>>> quadLineSeg;
	CVMat src2;
	src.copyTo(src2);
	for (int row = 0; row < quadRowNum; row++) {
		vector<vector<LineD>> vec_row;
		for (int col = 0; col < quadColNum; col++) {
			CoordinateDouble topLeft = mesh[row][col];
			CoordinateDouble topRight = mesh[row][col + 1];
			CoordinateDouble bottomLeft = mesh[row + 1][col];
			CoordinateDouble bottomRight = mesh[row + 1][col + 1];

			vector<LineD> lineInQuad;
			for (int i = 0; i < lines.size(); i++) {
				LineD line = lines[i];
				CoordinateDouble p1(line.row1, line.col1);
				CoordinateDouble p2(line.row2, line.col2);
				//�ж������Ƿ�ֱ����ͬһ��Quad��
				bool p1InQuad = isInQuad(p1, topLeft, topRight, bottomLeft, bottomRight);
				bool p2InQuad = isInQuad(p2, topLeft, topRight, bottomLeft, bottomRight);
				if (p1InQuad && p2InQuad) {
					lineInQuad.push_back(line);
				}
				else if (p1InQuad) { //q1��quad�����һ������
					vector<CoordinateDouble> intersections = intersectionWithQuad(line, topLeft, topRight, bottomLeft, bottomRight);
					if (intersections.size() != 0) {
						LineD cutLine(p1, intersections[0]);
						lineInQuad.push_back(cutLine);
					}
				}
				else if (p2InQuad) { //q2��quad�����һ������
					vector<CoordinateDouble> intersections = intersectionWithQuad(line, topLeft, topRight, bottomLeft, bottomRight);
					if (intersections.size() != 0) {
						LineD cutLine(p2, intersections[0]);
						lineInQuad.push_back(cutLine);
					}
				}
				else { //����һ��quad���棬������������
					vector<CoordinateDouble> intersections = intersectionWithQuad(line, topLeft, topRight, bottomLeft, bottomRight);
					if (intersections.size() == 2) {
						LineD cutLine(intersections[0], intersections[1]);
						lineInQuad.push_back(cutLine);
					}
				}
			}
			vec_row.push_back(lineInQuad);
		}
		quadLineSeg.push_back(vec_row);
	}
	return quadLineSeg;
}

//���߶��������н�ά��������ƽ��һά����
void flatten(vector<vector<vector<LineD>>> lineSeg, vector<LineD>& line_vec, Config config) {
	int quadRowNum = config.meshQuadRow;
	int quadColNum = config.meshQuadCol;
	for (int row = 0; row < quadRowNum; row++) {
		for (int col = 0; col < quadColNum; col++) {
			for (int k = 0; k < lineSeg[row][col].size(); k++) {
				line_vec.push_back(lineSeg[row][col][k]);
			}
		}
	}
}

//��ʼ���߶�
vector<vector<vector<LineD>>> initLineSeg(CVMat src, CVMat mask, Config config, vector<LineD>& lineSeg_flatten, vector<vector<CoordinateDouble>> mesh, vector<pair<int, double>>& id_theta, vector<double>& rotate_theta) {
	double thetaPerBin = PI / 49;
	//��Ե���ͣ���ֹ��⵽��Ե��
	reviseMask4Lines(mask);
	//lsd����߶�
	vector<LineD> lines = lsdDetect(src, mask);
	//���߶η��䵽ÿ��������
	vector<vector<vector<LineD>>> lineSeg = segmentLineInQuad(src, lines, mesh, config);

	flatten(lineSeg, lineSeg_flatten, config);

	for (int i = 0; i < lineSeg_flatten.size(); i++) {
		LineD line = lineSeg_flatten[i];
		//����б������arctan����theta��
		double theta = atan((line.row2 - line.row1) / (line.col2 - line.col1));
		//ԭʼ��������[-pi/2,pi/2)����ͨ����pi/2�任��[0,pi)
		int lineSegmentBucket = (int)round((theta + PI / 2) / thetaPerBin);
		assert(lineSegmentBucket < 50);
		//Ϊÿ���߶����ϽǶ�Bin��źͽǶ�
		id_theta.push_back(make_pair(lineSegmentBucket, theta));
		//��ʼ�������Բ��ü���ת��
		rotate_theta.push_back(0);
	}
	return lineSeg;
}

//��ԭʼ����origin�Ļ��������������Ӿ���addin
SparseMatrixDRow blockDiag(SparseMatrixDRow origin, MatrixXd addin, int quadId, Config config) {
	//��ʹ�õ�ʱ��addin�Ǽ���line energy���õ���C*��startWMat - endWMat�������Ǹ�2x8�ľ���
	int colsTotal = 8 * config.meshQuadRow * config.meshQuadCol;
	SparseMatrixDRow res(origin.rows() + addin.rows(), colsTotal);
	res.topRows(origin.rows()) = origin;

	int leftTopRow = origin.rows();
	int leftTopCol = 8 * quadId;
	//��origin������·��ı��ΪquadId��quad���addin�е�Ԫ��
	for (int row = 0; row < addin.rows(); row++) {
		for (int col = 0; col < addin.cols(); col++) {
			res.insert(leftTopRow + row, leftTopCol + col) = addin(row, col);
		}
	}
	/*
	[ [                  ]                                      ]
	| [                  ]                                      |
	|                    [                ]                     |
	|                    [                ]                     |
	...                                 .                      ...
	|                                     .                     |
	|                                       .                   |
	|                                          [              ] |
	[                                          [              ] ]
	*/

	res.makeCompressed();
	return res;
}


//Ȩ�ػ�Ϊ2x8�ľ��������ҳ�Vq���Եõ���quad������߶ζ˵��˫���Բ�ֵ�Ľ��
MatrixXd BiLinearWeights2Matrix(myBilinearWeights w) {
	MatrixXd mat(2, 8);
	double v1w = 1 - w.u - w.v + w.u * w.v;
	double v2w = w.u - w.u * w.v;
	double v3w = w.v - w.u * w.v;
	double v4w = w.u * w.v;
	mat << v1w, 0, v2w, 0, v3w, 0, v4w, 0,
		0, v1w, 0, v2w, 0, v3w, 0, v4w;
	return mat;
}

MatrixXd BiLinearWeights2Matrix(BilinearWeights w) {
	MatrixXd mat(2, 8);
	double v1w = 1 - w.s - w.t + w.s * w.t;
	double v2w = w.s - w.s * w.t;
	double v3w = w.t - w.s * w.t;
	double v4w = w.s * w.t;
	mat << v1w, 0, v2w, 0, v3w, 0, v4w, 0,
		0, v1w, 0, v2w, 0, v3w, 0, v4w;
	return mat;
}


//����a.x*b.y-a.y-b.x
double cross(CoordinateDouble a, CoordinateDouble b) {
	return a.col * b.row - a.row * b.col;
}

/*
��������getMyBilinearWeights
����˫���Բ�ֵ�������в�ֵ����
*/
myBilinearWeights getMyBilinearWeights(CoordinateDouble point, CoordinateInt upperLeftIndices, vector<vector<CoordinateDouble>> mesh) {

	//ԭ��https://www.iquilezles.org/www/articles/ibilinear/ibilinear.htm
	CoordinateDouble a = mesh[upperLeftIndices.row][upperLeftIndices.col];
	CoordinateDouble b = mesh[upperLeftIndices.row][upperLeftIndices.col + 1];
	CoordinateDouble d = mesh[upperLeftIndices.row + 1][upperLeftIndices.col];
	CoordinateDouble c = mesh[upperLeftIndices.row + 1][upperLeftIndices.col + 1];

	//E F G H
	CoordinateDouble e = b - a;
	CoordinateDouble f = d - a;
	CoordinateDouble g = a - b + c - d;
	CoordinateDouble h = point - a;

	//k2,k1,k0
	double k2 = cross(g, f);
	double k1 = cross(e, f) + cross(h, g);
	double k0 = cross(h, e);

	double u, v;
	if ((int)k2 == 0) {
		v = -k0 / k1;
		u = (h.col - f.col * v) / (e.col + g.col * v);
	}
	else {
		double w = k1 * k1 - 4.0 * k0 * k2;
		assert(w >= 0);
		w = sqrt(w);
		double v1 = (-k1 - w) / (2.0 * k2);
		double u1 = (h.col - f.col * v1) / (e.col + g.col * v1);

		double v2 = (-k1 + w) / (2.0 * k2);
		double u2 = (h.col - f.col * v2) / (e.col + g.col * v2);

		u = u1;
		v = v1;

		if (v < 0.0 || v>1.0 || u < 0.0 || u>1.0) {
			u = u2;
			v = v2;
		}
	}
	return myBilinearWeights(u, v);
}

/*
��������getBilinearWeights
�Ƕ��ڲ������ı����еĵ��˫���Բ�ֵ����
*/


//��ȡ˫���Բ�ֵ��Ȩ��
BilinearWeights getBilinearWeights(CoordinateDouble point, CoordinateInt upperLeftIndices, vector<vector<CoordinateDouble>> mesh) {
	//������ϣ����ϣ����£����µ������
	CoordinateDouble p0 = mesh[upperLeftIndices.row][upperLeftIndices.col];
	CoordinateDouble p1 = mesh[upperLeftIndices.row][upperLeftIndices.col + 1];
	CoordinateDouble p2 = mesh[upperLeftIndices.row + 1][upperLeftIndices.col];
	CoordinateDouble p3 = mesh[upperLeftIndices.row + 1][upperLeftIndices.col + 1];

	//����������ߵ�б��
	double topSlope = (p1.row - p0.row) / (p1.col - p0.col);
	double bottomSlope = (p3.row - p2.row) / (p3.col - p2.col);
	double leftSlope = (p2.row - p0.row) / (p2.col - p0.col);
	double rightSlope = (p3.row - p1.row) / (p3.col - p1.col);

	double quadRaticEpsilon = 0.01;
	if (topSlope == bottomSlope && leftSlope == rightSlope) {
		Matrix2d mat1;
		mat1 << p1.col - p0.col, p2.col - p0.col,
			p1.row - p0.row, p2.row - p0.row;
		MatrixXd mat2(2, 1);
		mat2 << point.col - p0.col, point.row - p0.row;

		MatrixXd matsolution = mat1.inverse() * mat2;
		BilinearWeights weights;
		weights.s = matsolution(0, 0);
		weights.t = matsolution(1, 0);
		return weights;
	}
	else if (leftSlope == rightSlope) {
		double a = (p1.col - p0.col) * (p3.row - p2.row) - (p1.row - p0.row) * (p3.col - p2.col);
		double b = point.row * ((p3.col - p2.col) - (p1.col - p0.col)) - point.col * ((p3.row - p2.row) - (p1.row - p0.row)) + p0.col * (p3.row - p2.row) - p0.row * (p3.col - p2.col) + p2.row * (p1.col - p0.col) - p2.col * (p1.row - p0.row);
		double c = point.row * (p2.col - p0.col) - point.col * (p2.row - p0.row) + p0.col * p2.row - p2.col * p0.row;

		double s1 = (-1 * b + sqrt(b * b - 4 * a * c)) / (2 * a);
		double s2 = (-1 * b - sqrt(b * b - 4 * a * c)) / (2 * a);
		double s;
		if (s1 >= 0 && s1 <= 1) {
			s = s1;
		}
		else if (s2 >= 0 && s2 <= 1) {
			s = s2;
		}
		else {
			if ((s1 > 1 && s1 - quadRaticEpsilon < 1) || (s2 > 1 && s2 - quadRaticEpsilon < 1)) {
				s = 1;
			}
			else if ((s1 < 0 && s1 + quadRaticEpsilon > 0) || (s2 < 0 && s2 + quadRaticEpsilon > 0)) {
				s = 0;
			}
			else {
				s = 0;
			}
		}
		double val = p2.row + (p3.row - p2.row) * s - p0.row - (p1.row - p0.row) * s;
		double t = (point.row - p0.row - (p1.row - p0.row) * s) / val;
		double valEpsilon = 0.1;
		if (fabs(val) < valEpsilon) {
			t = (point.col - p0.col - (p1.col - p0.col) * s) / (p2.col + (p3.col - p2.col) * s - p0.col - (p1.col - p0.col) * s);
		}
		BilinearWeights weights;
		weights.s = s;
		weights.t = t;
		return weights;
	}
	else {
		double a = (p2.col - p0.col) * (p3.row - p1.row) - (p2.row - p0.row) * (p3.col - p1.col);
		double b = point.row * ((p3.col - p1.col) - (p2.col - p0.col)) - point.col * ((p3.row - p1.row) - (p2.row - p0.row)) + (p2.col - p0.col) * (p1.row) - (p2.row - p0.row) * (p1.col) + (p0.col) * (p3.row - p1.row) - (p0.row) * (p3.col - p1.col);
		double c = point.row * (p1.col - p0.col) - (point.col) * (p1.row - p0.row) + p0.col * p1.row - p1.col * p0.row;
		double t1 = (-1 * b + sqrt(b * b - 4 * a * c)) / (2 * a);
		double t2 = (-1 * b - sqrt(b * b - 4 * a * c)) / (2 * a);
		double t;
		if (t1 >= 0 && t1 <= 1) {
			t = t1;
		}
		else if (t2 >= 0 && t2 <= 1) {
			t = t2;
		}
		else {
			if ((t1 > 1 && t1 - quadRaticEpsilon < 1) || (t2 > 1 && t2 - quadRaticEpsilon < 1)) {
				t = 1;
			}
			else if ((t1 < 0 && t1 + quadRaticEpsilon > 0) || (t2 < 0 && t2 + quadRaticEpsilon > 0)) {
				t = 0;
			}
			else {
				t = 0;
			}
		}

		double val = p1.row + (p3.row - p1.row) * t - p0.row - (p2.row - p0.row) * t;
		double s = (point.row - p0.row - (p2.row - p0.row) * t) / val;
		double valEpsilon = 0.1;
		if (fabs(val) < valEpsilon) {
			s = (point.col - p0.col - (p2.col - p0.col) * t) / (p1.col + (p3.col - p1.col) * t - p0.col - (p2.col - p0.col) * t);
		}
		BilinearWeights weights;
		weights.s = clamp(s, 0, 1);
		weights.t = clamp(t, 0, 1);
		return weights;
	}
}


//Line Preservation Energy��ϵ���ľ���
SparseMatrixDRow getLineMat(CVMat src, CVMat mask, vector<vector<CoordinateDouble>> mesh, vector<double> rotate_theta, vector<vector<vector<LineD>>> lineSeg, vector<pair<MatrixXd, MatrixXd>>& BilinearVec, Config config, int& lineNum, vector<bool>& bad) {
	int lineTmpNum = -1;
	int rows = config.rows;
	int cols = config.cols;
	int quadRowNum = config.meshQuadRow;
	int quadColNum = config.meshQuadCol;

	SparseMatrixDRow energyLine;
	for (int row = 0; row < quadRowNum; row++) {
		for (int col = 0; col < quadColNum; col++) {
			//��ȡ��ǰ�������е��߶�
			vector<LineD> lineSegInQuad = lineSeg[row][col];
			//��¼��ǰ����ID
			int quadId = row * quadColNum + col;
			if (lineSegInQuad.size() == 0) {
				continue;
			}
			else {
				CoordinateInt topLeft(row, col);
				MatrixXd cRowStack(0, 8);
				for (int k = 0; k < lineSegInQuad.size(); k++) {
					lineTmpNum++;
					LineD line = lineSegInQuad[k];
					CoordinateDouble lineStart(line.row1, line.col1);
					CoordinateDouble lineEnd(line.row2, line.col2);

					//myBilinearWeights startWeight = getMyBilinearWeights(lineStart, topLeft, mesh); //��˫����
					BilinearWeights startWeight = getBilinearWeights(lineStart, topLeft, mesh);  //˫����
					MatrixXd startWMat = BiLinearWeights2Matrix(startWeight);
					//myBilinearWeights endWeight = getMyBilinearWeights(lineEnd, topLeft, mesh);
					BilinearWeights endWeight = getBilinearWeights(lineEnd, topLeft, mesh);
					MatrixXd endWMat = BiLinearWeights2Matrix(endWeight);
					VectorXd S = getVertice(row, col, mesh); // Vq 8 x 1 
					//startWMat * S��������˫���Բ�ֵ����߶���ʼ�������
					Vector2d ans = startWMat * S - Vector2d(lineStart.col, lineStart.row); //2 x 1
					Vector2d ans2 = endWMat * S - Vector2d(lineEnd.col, lineEnd.row);
					//���������
					if (ans2.norm() >= 0.0001 || ans.norm() >= 0.0001) {
						bad.push_back(true);
						BilinearVec.push_back(make_pair(MatrixXd::Zero(2, 8), MatrixXd::Zero(2, 8)));
						continue;
					}
					assert(ans.norm() < 0.0001);
					assert(ans2.norm() < 0.0001);
					bad.push_back(false);

					double theta = rotate_theta[lineTmpNum];
					BilinearVec.push_back(make_pair(startWMat, endWMat));

					Matrix2d R;
					R << cos(theta), -sin(theta),
						sin(theta), cos(theta);
					MatrixXd ehat(2, 1);
					ehat << line.col1 - line.col2, line.row1 - line.row2;
					MatrixXd tmp = (ehat.transpose() * ehat).inverse();
					Matrix2d I = Matrix2d::Identity();
					MatrixXd C = R * ehat * tmp * (ehat.transpose()) * (R.transpose()) - I; //2x2
					//startWMat - endWMat��2x8�ľ���
					MatrixXd CT = C * (startWMat - endWMat); //(startWMat - endWMat) * S = e
					cRowStack = VStack(cRowStack, CT);
				}
				energyLine = blockDiag(energyLine, cRowStack, quadId, config);
			}
		}
	}
	lineNum = lineTmpNum;
	return energyLine;
}



////����thetaֵ
//vector<vector<vector<pair<int, double>>>> calcTheta(vector<vector<vector<Line_rotate>>> lineSeg, Config config) {
//
//	vector<vector<vector<pair<int, double>>>> lineGroup;
//	for (int row = 0; row < config.meshQuadRow; row++) {
//		vector<vector<pair<int, double>>> row_vec;
//		for (int col = 0; col < config.meshQuadCol; col++) {
//			vector<pair<int, double>> vec;
//			row_vec.push_back(vec);
//		}
//		lineGroup.push_back(row_vec);
//	}
//	double qstep = PI / 49;
//	for (int row = 0; row < config.meshQuadRow; row++) {
//		for (int col = 0; col < config.meshQuadCol; col++) {
//			vector<Line_rotate> linevec = lineSeg[row][col];
//			int lineNum = (int)linevec.size();
//			for (int i = 0; i < lineNum; i++) {
//				Line_rotate line = linevec[i];
//				Vector2d pstart = line.pstart;
//				Vector2d pend = line.pend;
//				double theta = atan((pstart(0) - pend(0)) / (pstart(1) - pend(1)));
//				int groupId = (int)(round((theta + PI / 2) / qstep) + 1);
//				lineGroup[row][col].push_back(make_pair(groupId, theta));
//			}
//		}
//	}
//	return lineGroup;
//}

//�޲�
CVMat fillMissingPixel(CVMat& img, const CVMat mask)
{
	//rowΪy�ᣬcolΪx��
	assert(img.rows == mask.rows);
	assert(img.cols == mask.cols);
	CVMat mask_2;
	int size_erode = 3;
	CVMat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size_erode, size_erode));
	cv::erode(mask, mask_2, element);//255�Ƿ�ͼ�Ĳ��֣�����ʴ��һЩ���
	/*cv::imshow("temp", mask_2);
	cv::imshow("temp2", mask);
	cv::waitKey(0);	*/
	for (int row = 0; row < mask.rows; row++)
	{
		for (int col = 0; col < mask.cols; col++)
		{
			if (mask.at<uchar>(row, col) == 255 && mask_2.at<uchar>(row, col) == 0)
			{
				for (int i = 0; i < size_erode; i++)
				{
					int temp_y = row - 2 + i / size_erode;
					int temp_x = col - 2 + i % size_erode;
					if (temp_y >= 0 && temp_y <= mask.rows && temp_x >= 0 && temp_x <= mask.cols)
					{
						if (mask.at<uchar>(temp_y, temp_x) == 0)
						{
							img.at<cv::Vec3b>(row, col) = img.at<cv::Vec3b>(temp_y, temp_x);
							break;
						}
					}
				}
			}
		}
	}
	return mask_2;
}