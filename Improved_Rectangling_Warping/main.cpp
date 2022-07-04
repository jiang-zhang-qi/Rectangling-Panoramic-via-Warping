
#include "LocalWarp.h"
#include "ComLib.h"
#include "GlobalWarp.h"
#include "lsd.h"
#define GLUT_DISABLE_ATEXIT_HACK
#include <GL/glut.h>
#include <cmath>
#include <iomanip>
#define WindowTitle "OpenGL纹理测试"
GLuint texGround;
vector<vector<CoordinateDouble>> outputmesh;
vector<vector<CoordinateDouble>> mesh;
CVMat image;
bool flag_display = true;
double avgScaledX = 1, avgScaledY = 1;
bool factorJudgement = false;
double factor;
int viewS = 2;

GLuint mat2Texture(CVMat mat, GLenum minFilter = GL_LINEAR, GLenum magFilter = GL_LINEAR, GLenum wrapFilter = GL_REPEAT) {

	GLuint textureId;
	glGenTextures(1, &textureId);

	glBindTexture(GL_TEXTURE_2D, textureId);

	if (magFilter == GL_LINEAR_MIPMAP_LINEAR || magFilter == GL_LINEAR_MIPMAP_NEAREST || magFilter == GL_NEAREST_MIPMAP_LINEAR || magFilter == GL_NEAREST_MIPMAP_NEAREST) {
		cout << "sill error" << endl;
		magFilter = GL_LINEAR;
	}
	//设置纹理插值方式
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);

	GLenum inputColourFormat = GL_BGR_EXT;
	if (mat.channels() == 1) {
		inputColourFormat = GL_LUMINANCE;
	}
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mat.cols, mat.rows, 0, inputColourFormat, GL_UNSIGNED_BYTE, mat.ptr());
	return textureId;
}

void display(void) {
	//清除屏幕
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//加载纹理数据
	texGround = mat2Texture(image);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, image.cols, image.rows, 0);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texGround);
	//遍历每个网格
	if (flag_display) {
		for (int row = 0; row < 20; row++) {
			for (int col = 0; col < 20; col++) {
				CoordinateDouble& coord = outputmesh[row][col];
				CoordinateDouble& localcoord = mesh[row][col];
				localcoord.row /= image.rows;
				localcoord.col /= image.cols;
			}
		}
		flag_display = false;
	}
	for (int row = 0; row < 19; row++) {
		for (int col = 0; col < 19; col++) {
			CoordinateDouble local_left_top = mesh[row][col];
			CoordinateDouble local_right_top = mesh[row][col + 1];
			CoordinateDouble local_left_bottom = mesh[row + 1][col];
			CoordinateDouble local_right_bottom = mesh[row + 1][col + 1];

			CoordinateDouble global_left_top = outputmesh[row][col];
			CoordinateDouble global_right_top = outputmesh[row][col + 1];
			CoordinateDouble global_left_bottom = outputmesh[row + 1][col];
			CoordinateDouble global_right_bottom = outputmesh[row + 1][col + 1];

			//glTexCoord2d用于设置原mesh上的四个坐标，而glVertex2d则用于设置输出mesh上的四个坐标
			glBegin(GL_QUADS);
			glTexCoord2d(local_left_top.col, local_left_top.row); glVertex2d(global_left_top.col, global_left_top.row);
			glTexCoord2d(local_right_top.col, local_right_top.row); glVertex2d(global_right_top.col, global_right_top.row);
			glTexCoord2d(local_right_bottom.col, local_right_bottom.row); glVertex2d(global_right_bottom.col, global_right_bottom.row);
			glTexCoord2d(local_left_bottom.col, local_left_bottom.row);	glVertex2d(global_left_bottom.col, global_left_bottom.row);
			glEnd();
		}
	}
	glDisable(GL_TEXTURE_2D);
	glutSwapBuffers();
}


int main(int argc, char* argv[]) {
	cout << "Rectangling!" << endl;
	//读取图片
	//image = myimReadfun("D:\\test\\3_input.jpeg", factor, 0.38, factorJudgement, true, viewS, 2);

	image = myimReadfun("D:\\test\\4_input.png", factor, 1, factorJudgement, false, viewS, 3);

	//image = myimReadfun("D:\\test\\2_input.jpeg", factor, 0.9, factorJudgement, false, viewS, 1);

	//image = cv::imread("D:\\test\\4_input.png");


	//计时
	double Time = (double)cv::getTickCount();

	CVMat scaled_img;
	//DownSampled
	cv::resize(image, scaled_img, cv::Size(0, 0), factor, factor, cv::INTER_AREA);

	//cv::namedWindow("downsampled Image", cv::WINDOW_AUTOSIZE);
	//cv::imshow("downsampled Image", scaled_img);
	//cv::waitKey(0);

	Config config(scaled_img.rows, scaled_img.cols, 20, 20);
	CVMat mask = MaskContour(scaled_img);

	CVMat tmpMask;
	mask.copyTo(tmpMask);
	CVMat wrapped_img = CVMat::zeros(scaled_img.size(), CV_8UC3);
	//Localwarp
	vector<vector<CoordinateInt>> displacement = localWarp(scaled_img, wrapped_img, tmpMask);

	cout << "Localwarp：" << ((double)cv::getTickCount() - Time) / cv::getTickFrequency() << "s!" << endl;

	//生成wrap后图像的网格
	mesh = getRectangleMesh(scaled_img, config);
	//把mesh逆回去
	warpMeshBack(mesh, displacement, config);

	//对角线上存放 shape energy的系数矩阵
	SparseMatrixDRow shapeEnergy = getShapeMat(mesh, config);
	//存放标志着Vq的点的矩阵，实际上的用处是用来做行变换的！！
	SparseMatrixDRow Q = getVertex2ShapeMat(mesh, config);
	//将边界的点做标记，左右边界点固定x=1，上下边界点固定y=1以此标识
	pair<SparseMatrixDRow, VectorXd> pair_dvec_B = getBoundaryMat(scaled_img, mesh, config);
	//初始化线段以及对应角度
	vector<pair<int, double>> id_theta;
	vector<LineD> line_flatten;
	vector<double> rotate_theta;
	vector<vector<vector<LineD>>> LineSeg = initLineSeg(scaled_img, mask, config, line_flatten, mesh, id_theta, rotate_theta);



	for (int iter = 1; iter <= 10; iter++) {
		cout << iter << endl;
		//update V
		int Nl = 0;
		vector<pair<MatrixXd, MatrixXd>> BilinearVec;
		vector<bool> bad;
		//得到对角线上存放line energy的系数矩阵
		SparseMatrixDRow lineEnergy = getLineMat(scaled_img, mask, mesh, rotate_theta, LineSeg, BilinearVec, config, Nl, bad);
		double Nq = config.meshQuadRow * config.meshQuadCol;
		double lambdaB = INF;
		double lambdaL = 100;
		//这里乘上Q就是为了做行变换，其实可以连在后面一起看就是shapeEnergy * (Q * x)!!
		SparseMatrixDRow shape = (1 / sqrt(Nq)) * (shapeEnergy * Q);
		SparseMatrixDRow boundary = sqrt(lambdaB) * pair_dvec_B.first;
		SparseMatrixDRow line = sqrt((lambdaL / Nl)) * (lineEnergy * Q);
		/*
			实际上是要求二元函数(b-K2*x)'(b-K2*x)的最小，现在转化成了最小二乘的最小
			即希望 K2*x = b成立，而非方阵故用最小二乘 K2'*K2*x = k2'*b
			由于K2'*K2是对称正定矩阵我们用Cholesky解法求解
		*/
		SparseMatrixDRow K = VStack(shape, line);
		SparseMatrixDRow K2 = VStack(K, boundary);
		SparseMatrixD K2_trans = K2.transpose();
		SparseMatrixD A = K2_trans * K2;

		VectorXd B = pair_dvec_B.second;
		VectorXd BA = VectorXd::Zero(K2.rows());
		//tail(n)取向量尾部的n个元素
		BA.tail(B.size()) = sqrt(lambdaB) * B;
		VectorXd b = K2_trans * BA;
		//这里的x相当于存放着3个mesh     x = [x0, y0, ..., x399, y399, x0, y0, ..., x399, y399, x0, y0, ..., x399, y399]^T
		VectorXd x;
		CSolve* p_A = new CSolve(A);
		x = p_A->solve(b);


		//update theta
		outputmesh = vector2mesh(x, config);
		int tmpLineNum = -1;
		VectorXd thetaGroup = VectorXd::Zero(50);
		VectorXd thetaGrouCnt = VectorXd::Zero(50);

		for (int row = 0; row < config.meshQuadRow; row++) {
			for (int col = 0; col < config.meshQuadCol; col++) {
				vector<LineD> lineSegInQuad = LineSeg[row][col];
				int quadId = row * config.meshQuadCol + col;
				if (lineSegInQuad.size() == 0) {
					continue;
				}
				else {
					VectorXd S = getVertice(row, col, outputmesh);
					for (int k = 0; k < lineSegInQuad.size(); k++) {
						tmpLineNum++;
						if (bad[tmpLineNum] == true) {
							continue;
						}
						pair<MatrixXd, MatrixXd> Bstartend = BilinearVec[tmpLineNum];
						MatrixXd startWMat = Bstartend.first;
						MatrixXd endWMat = Bstartend.second;
						Vector2d newstart = startWMat * S;
						Vector2d newend = endWMat * S;

						double theta = atan((newstart(1) - newend(1)) / (newstart(0) - newend(0)));
						double deltatheta = theta - id_theta[tmpLineNum].second;
						if (isnan(id_theta[tmpLineNum].second) || isnan(deltatheta)) {
							continue;
						}
						if (deltatheta > (PI / 2)) {
							deltatheta -= PI;
						}
						if (deltatheta < -(PI / 2)) {
							deltatheta += PI;
						}
						thetaGroup(id_theta[tmpLineNum].first) += deltatheta;
						thetaGrouCnt(id_theta[tmpLineNum].first) += 1;
					}
				}
			}
		}
		//计算theta均值
		for (int ii = 0; ii < thetaGroup.size(); ii++) {
			thetaGroup(ii) /= thetaGrouCnt(ii);
		}
		//更新rotate_theta,就是e和e_hat之间的夹角
		for (int ii = 0; ii < rotate_theta.size(); ii++) {
			rotate_theta[ii] = thetaGroup[id_theta[ii].first];  //第ii根线的旋转角角等于该线角度所属bin的旋转角均值
		}
	}
	//CVMat erode_mask;
	//erode_mask = fillMissingPixel(image, mask);//修补像素
	//erode_mask = fillMissingPixel(image, erode_mask);//修补像素

	//calculate x and y scaled factor
	if (factorJudgement) {
		computeScaling(avgScaledX, avgScaledY, mesh, outputmesh, config);
		cout << avgScaledX << " " << avgScaledY << endl;
		enlargeMesh(mesh, 1 / avgScaledX, 1 / avgScaledY, config);
		enlargeMesh(outputmesh, 1 / avgScaledX, 1 / avgScaledY, config);
		cv::resize(image, image, cv::Size(0, 0), 1 / avgScaledX, 1 / avgScaledY);
	}

	//UpSample
	enlargeMesh(mesh, 1 / factor, 1 / factor, config);
	enlargeMesh(outputmesh, 1 / factor, 1 / factor, config);

	Time = (double)cv::getTickCount() - Time;
	cout << "程序所用时间：" << Time / cv::getTickFrequency() << "s!" << endl;

	//OpenGL贴图加速
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	//glutInitWindowSize(image.cols, image.rows);
	glutInitWindowSize(image.cols / viewS, image.rows / viewS);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Rectangling Panoramic Image");
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(&display);
	glutMainLoop();
	system("pause");
	return 0;
}

