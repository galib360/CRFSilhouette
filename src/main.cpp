#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>
#include <omp.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>

#include "densecrf.h"
#include <cstdio>
#include <cmath>
#include "util.h"
#include "common.h"

using namespace cv;
using namespace dnn;
using namespace std;
using namespace std::chrono;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float maskThreshold = 0.75; // Underestimation mask; more, the smaller the mask
float maskThreshold2 = 0.15; // Overestimation mask; less, the bigger the mask
int nFrames = 31; //Number of frames in a set
int view = 8; //Number of sets
int startView = 0; //Starting set, will run until it reaches the number of sets(View), Has to have same no. of frames
int startFrame = 0;
int rectPad = 10;
float R = 0, P = 0, F = 0, A = 0; //Final average evaluations
//Rect BB;
string inputdir = "Costume3Walk";
ofstream myfile;
static int docrf = 1;
static int doeval = 1;
static int domorpho = 0;
const float GT_PROB = 0.5;


////For manipulating brightness and contrast
//float alpha = 2.2;
//int beta = 20;

vector<string> classes;
vector<Scalar> colorsmcnn;

// Postprocess the neural network's output for each frame
Mat postprocess(Mat& frame, const vector<Mat>& outs, int& countFrame,
		int& countView, Rect& BB);

MatrixXf computeUnary(const VectorXs & lbl, int M) {
	const float u_energy = -log(1.0 / M);
	const float n_energy = -log((1.0 - GT_PROB) / (M - 1));
	const float p_energy = -log(GT_PROB);
	MatrixXf r(M, lbl.rows());
	r.fill(u_energy);
	//printf("%d %d %d \n",im[0],im[1],im[2]);
	for (int k = 0; k < lbl.rows(); k++) {
		// Set the energy
		if (lbl[k] >= 0) {
			r.col(k).fill(n_energy);
			r(lbl[k], k) = p_energy;
		}
	}
	return r;
}

//Mat DCRF(Mat& frame, Mat& crfmerged) {
//
//	Mat crfoutput;
//	// Number of labels
//	const int M = 21;
//	// Load the color image and some crude annotations (which are used in a simple classifier)
//	int W = frame.cols;
//	int H = frame.rows;
//	int GW = crfmerged.cols;
//	int GH = crfmerged.rows;
//	unsigned char * im = frame.data;
//	if (!im) {
//		printf("Failed to load image!\n");
//		return crfoutput;
//	}
//	unsigned char * anno = crfmerged.data;
//	if (!anno) {
//		printf("Failed to load annotations!\n");
//		return crfoutput;
//	}
//	if (W != GW || H != GH) {
//		printf("Annotation size doesn't match image!\n");
//		return crfoutput;
//	}
//
//	/////////// Put your own unary classifier here! ///////////
//	MatrixXf unary = computeUnary(getLabeling(anno, W * H, M), M);
//	///////////////////////////////////////////////////////////
//
//	// Setup the CRF model
//	DenseCRF2D crf(W, H, M);
//	// Specify the unary potential as an array of size W*H*(#classes)
//	// packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
//	crf.setUnaryEnergy(unary);
//	// add a color independent term (feature = pixel location 0..W-1, 0..H-1)
//	// x_stddev = 3
//	// y_stddev = 3
//	// weight = 3
//	crf.addPairwiseGaussian(3, 3, new PottsCompatibility(3));
//	// add a color dependent term (feature = xyrgb)
//	// x_stddev = 60
//	// y_stddev = 60
//	// r_stddev = g_stddev = b_stddev = 20
//	// weight = 10
//	crf.addPairwiseBilateral(80, 80, 13, 13, 13, im,
//			new PottsCompatibility(10));
//
//	// Do map inference
//// 	MatrixXf Q = crf.startInference(), t1, t2;
//// 	printf("kl = %f\n", crf.klDivergence(Q) );
//// 	for( int it=0; it<5; it++ ) {
//// 		crf.stepInference( Q, t1, t2 );
//// 		printf("kl = %f\n", crf.klDivergence(Q) );
//// 	}
//// 	VectorXs map = crf.currentMap(Q);
//	VectorXs map = crf.map(5);
//	// Store the result
//	unsigned char *res = colorize(map, W, H);
//	crfoutput = Mat(H, W, CV_8UC3, res);
//
//	delete[] im;
//	delete[] anno;
//	delete[] res;
//
//	return crfoutput;
//}

int main() {

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	string resultdir = inputdir + "/result.txt";
	myfile.open(resultdir);
	myfile << inputdir << endl;
	myfile << "" << endl;

	// Load names of classes
	string classesFile = "mscoco_labels.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line))
		classes.push_back(line);

	// Load the colors
	string colorsFile = "colors.txt";
	ifstream colorFptr(colorsFile.c_str());
	while (getline(colorFptr, line)) {
		char* pEnd;
		double r, g, b;
		r = strtod(line.c_str(), &pEnd);
		g = strtod(pEnd, NULL);
		b = strtod(pEnd, NULL);
		Scalar color = Scalar(r, g, b, 255.0);
		colorsmcnn.push_back(Scalar(r, g, b, 255.0));
	}

	// Give the configuration and weight files for the model
	String textGraph = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
	String modelWeights = "frozen_inference_graph.pb";
	//String modelWeights = "saved_model.pb";

	// Load the network
//	Net net = readNetFromTensorflow(modelWeights, textGraph);
//	net.setPreferableBackend(DNN_BACKEND_OPENCV);
//	net.setPreferableTarget(DNN_TARGET_CPU);

//	// Open a video file or an image file or a camera stream.
//	string pathGT, pathData, outputFile;
//	Mat frame, blob, silGT;



	for (int countView = startView; countView < view; countView++) {
		float Rset = 0, Pset = 0, Fset = 0, Aset = 0;
		int countFrame = startFrame;
#pragma omp parallel num_threads(3)
#pragma omp for
		for (countFrame = startFrame; countFrame < nFrames; countFrame++) {
			string pathGT, pathData, outputFile;
			Mat frame, blob, silGT;
			Rect BB;
			Net net = readNetFromTensorflow(modelWeights, textGraph);
			net.setPreferableBackend(DNN_BACKEND_OPENCV);
			net.setPreferableTarget(DNN_TARGET_CPU);
			// Open a video file or an image file or a camera stream.

			//			pathData = "experiment/cam0" + to_string(countView) + "/"
			//					+ to_string(countFrame) + ".png";

			if (countView < 10) {
				pathData = inputdir + "/data/cam0" + to_string(countView) + "/"
						+ to_string(countFrame) + ".png";
				//for experiment:
				//				pathData = inputdir+"/data/cam0" + to_string(countView) + "/"
				//										+ to_string(countFrame) + ".jpg";

				pathGT = inputdir + "/Ground/cam0" + to_string(countView) + "/"
						+ to_string(countFrame) + ".png";
				//for dancer4D, below:
				//				pathGT = inputdir+"/Ground/cam0" + to_string(countView) + "/"
				//										+ to_string(countFrame) + ".pbm";

			} else if (countView >= 10) {
				pathData = inputdir + "/data/cam" + to_string(countView) + "/"
						+ to_string(countFrame) + ".png";

				pathGT = inputdir + "/Ground/cam" + to_string(countView) + "/"
						+ to_string(countFrame) + ".png";
				//for dancer4D, below:
				//				pathGT = inputdir+"/Ground/cam" + to_string(countView) + "/"
				//										+ to_string(countFrame) + ".pbm";

			}

			//			pathData = "dancer4D/data/cam0" + to_string(countView) + "/"
			//								+ to_string(countFrame) + ".png";
			//
			//			pathGT = "dancer4D/Ground/cam0" + to_string(countView) + "/"+ to_string(countFrame) + ".pbm";

			frame = imread(pathData);//original picture to estimate silhouette
			silGT = imread(pathGT, IMREAD_GRAYSCALE);//Ground truth silhouette

			//if resize needed, for large pictures
//			resize(frame, frame, cv::Size(), 0.5, 0.5);
//			resize(silGT, silGT, cv::Size(), 0.5, 0.5);

			//resize(frame, frame, cv::Size(), 0.5, 0.5);
			//frame.convertTo(frame, -1, alpha, beta);

			if (frame.empty()) {
				cout << "No picture found !!!" << endl;
				//break;
			}
			//			imshow("frame", frame);
			//			waitKey(0);
			blobFromImage(frame, blob, 1.0, Size(frame.cols, frame.rows),
					Scalar(), true, false);

			net.setInput(blob);

			// Runs the forward pass to get output from the output layers
			std::vector<String> outNames(2);
			outNames[0] = "detection_out_final";
			outNames[1] = "detection_masks";
			vector<Mat> outs;
			net.forward(outs, outNames);

			// Extract the bounding box, object mask and silhouettes
			Mat silEst = postprocess(frame, outs, countFrame, countView, BB);

			//Evaluation code goes here ------------>

			cv::threshold(silGT, silGT, 20, 255, cv::THRESH_BINARY);
			cv::threshold(silEst, silEst, 20, 255, cv::THRESH_BINARY);

			float tn = 0, fp = 0.0000001, fn = 0.0000001, tp = 0;
			float recall, precision, f1, accuracy;

			int x = BB.x;
			int y = BB.y;
			int height = BB.height;
			int width = BB.width;

			if (doeval == 1) {

				for (int row = y; row < y + height; row++) {
					for (int col = x; col < x + width; col++) {
						if (silGT.at<uchar>(row, col) == 0
								&& silEst.at<uchar>(row, col) != 0) {
							fp += 1;
						} else if (silGT.at<uchar>(row, col) != 0
								&& silEst.at<uchar>(row, col) == 0) {
							fn += 1;
						} else if (silGT.at<uchar>(row, col) != 0
								&& silEst.at<uchar>(row, col) != 0) {
							tp += 1;
						} else if (silGT.at<uchar>(row, col) == 0
								&& silEst.at<uchar>(row, col) == 0) {
							tn += 1;
						}
					}
				}
			}

			recall = tp / (tp + fn);
			precision = tp / (tp + fp);
			if (recall == 0 || precision == 0) {
				f1 = 0;
				myfile << "no detection in cam" << countView << " ,frame "
						<< countFrame << endl;
			} else {
				f1 = 2 * ((precision * recall) / (precision + recall));
			}

			accuracy = (tp + tn) / (tp + tn + fp + fn);

			frame.release();
			silGT.release();
			silEst.release();

			Rset += recall;
			Pset += precision;
			Fset += f1;
			Aset += accuracy;

		}
		Rset = Rset / nFrames;
		Pset = Pset / nFrames;
		Fset = Fset / nFrames;
		Aset = Aset / nFrames;

		cout << "for set0" << countView << ": " << endl;
		cout << "Recall: " << Rset << endl;
		cout << "Precision: " << Pset << endl;
		cout << "F1 Score: " << Fset << endl;
		cout << "Accuracy: " << Aset << endl;
		cout << " " << endl;

		myfile << "for set0" << countView << ": " << endl;
		myfile << "Recall: " << Rset << endl;
		myfile << "Precision: " << Pset << endl;
		myfile << "F1 Score: " << Fset << endl;
		myfile << "Accuracy: " << Aset << endl;
		myfile << " " << endl;

		R += Rset;
		P += Pset;
		F += Fset;
		A += Aset;

	}

	R = R / view;
	P = P / view;
	F = F / view;
	A = A / view;

	cout << "for all the sets: " << endl;
	cout << "Recall: " << R << endl;
	cout << "Precision: " << P << endl;
	cout << "F1 Score: " << F << endl;
	cout << "Accuracy: " << A << endl;
	cout << " " << endl;

	myfile << "for all the sets: " << endl;
	myfile << "Recall: " << R << endl;
	myfile << "Precision: " << P << endl;
	myfile << "F1 Score: " << F << endl;
	myfile << "Accuracy: " << A << endl;
	myfile << " " << endl;
	myfile.close();

	//	delete[] im;
	//	delete[] anno;

	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	auto durationms = duration_cast<milliseconds>(t2 - t1).count();
	auto durations = duration_cast<seconds>(t2 - t1).count();
	auto durationmin = duration_cast<minutes>(t2 - t1).count();
	cout << "Total code execution time: " << durationms << " milliseconds"
			<< endl;
	cout << "Total code execution time: " << durations << " seconds" << endl;
	cout << "Total code execution time: " << durationmin << " minutes" << endl;

	return 1;

}

Mat postprocess(Mat& frame, const vector<Mat>& outs, int& countFrame,
		int& countView, Rect& BB) {
	Mat outDetections = outs[0];
	Mat outMasks = outs[1];

	// Output size of masks is NxCxHxW where
	// N - number of detected boxes
	// C - number of classes (excluding background)
	// HxW - segmentation shape
	const int numDetections = outDetections.size[2];
	const int numClasses = outMasks.size[1];

	outDetections = outDetections.reshape(1, outDetections.total() / 7);
//    cout<<"Number of detection: "<<numDetections<<endl;
	for (int i = 0; i < 1; ++i) //only taking one detection for all detection, i<numDetections
			{
		float score = outDetections.at<float>(i, 2);
		if (score > confThreshold) {
			// Extract the bounding box
			int classId = static_cast<int>(outDetections.at<float>(i, 1));
			int left = static_cast<int>(frame.cols
					* outDetections.at<float>(i, 3));
			int top = static_cast<int>(frame.rows
					* outDetections.at<float>(i, 4));
			int right = static_cast<int>(frame.cols
					* outDetections.at<float>(i, 5));
			int bottom = static_cast<int>(frame.rows
					* outDetections.at<float>(i, 6));

			left = max(0, min(left, frame.cols - 1)) - rectPad;
			top = max(0, min(top, frame.rows - 1)) - rectPad;
			right = max(0, min(right, frame.cols - 1)) + rectPad;
			bottom = max(0, min(bottom, frame.rows - 1)) + rectPad;
			Rect box = BB = Rect(left, top, right - left + 1, bottom - top + 1);

			// Extract the mask for the object
			Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F,
					outMasks.ptr<float>(i, classId));    //small
			Mat objectMask2(outMasks.size[2], outMasks.size[3], CV_32F,
					outMasks.ptr<float>(i, classId));    //big
			resize(objectMask, objectMask, Size(box.width, box.height));
			resize(objectMask2, objectMask2, Size(box.width, box.height));
			Mat mask = (objectMask > maskThreshold);
			Mat mask2 = (objectMask2 > maskThreshold2);

			//for CRF
			Mat crf(frame.rows, frame.cols, CV_8UC3);
			Mat crf2(frame.rows, frame.cols, CV_8UC3);
			crf.setTo(cv::Scalar(0, 200, 0));
			crf2.setTo(cv::Scalar(0, 200, 0));

			Mat maskcrf(mask.rows, mask.cols, CV_8UC3);
			Mat maskcrf2(mask2.rows, mask2.cols, CV_8UC3);
			for (int r = 0; r < maskcrf.rows; r++) {
				for (int c = 0; c < maskcrf.cols; c++) {
					if (mask.at<uchar>(r, c) == 0) {
						maskcrf.at<Vec3b>(r, c) = Vec3b(0, 200, 0);
					}
					if (mask2.at<uchar>(r, c) == 0) {
						maskcrf2.at<Vec3b>(r, c) = Vec3b(0, 200, 0);
					}
					if (mask.at<uchar>(r, c) != 0) {
						maskcrf.at<Vec3b>(r, c) = Vec3b(0, 50, 0);
					}
					if (mask2.at<uchar>(r, c) != 0) {
						maskcrf2.at<Vec3b>(r, c) = Vec3b(0, 0, 0);
					}
				}
			}

//			imshow("maskcrf",maskcrf);
//			imshow("maskcrf2",maskcrf2);
//			waitKey(0);

			try {
				maskcrf2.copyTo(crf2(box));
				maskcrf.copyTo(crf(box));
			} catch (Exception e) {
				cout << "could not copy box to crf frame" << endl;
				Mat mask_fgpf(frame.size(), CV_8UC1, Scalar(0));
				string outputPath;
				if (countView < 10) {
					outputPath = inputdir + "/Crf/cam0" + to_string(countView)
							+ "/" + to_string(countFrame) + ".png";
				} else if (countView >= 10) {
					outputPath = inputdir + "/Crf/cam" + to_string(countView)
							+ "/" + to_string(countFrame) + ".png";
				}
				imwrite(outputPath, mask_fgpf);
				return mask_fgpf;
			}

//			imshow("crf", crf);
//			imshow("crf2", crf2);
//			waitKey(0);

//			addWeighted( crf, 0.5, crf2, 0.5, 0.0, crf);
			Mat crfmerged;
			addWeighted(crf, 1, crf2, 2, 0.0, crfmerged);

			for (int r = 0; r < crfmerged.rows; r++) {
				for (int c = 0; c < crfmerged.cols; c++) {
					if (crfmerged.at<Vec3b>(r, c) == Vec3b(0, 200, 0)) {
						crfmerged.at<Vec3b>(r, c) = Vec3b(0, 0, 0);
					}
				}
			}

			imwrite("crfanno.png", crfmerged);

			Mat crfoutput;
			// Number of labels
			const int M = 21;
			// Load the color image and some crude annotations (which are used in a simple classifier)
			int W = frame.cols;
			int H = frame.rows;
			int GW = crfmerged.cols;
			int GH = crfmerged.rows;
//			unsigned char * im = frame.data;
//			if (!im) {
//				printf("Failed to load image!\n");
//				return crfoutput;
//			}
//			unsigned char * anno = crfmerged.data;
//			if (!anno) {
//				printf("Failed to load annotations!\n");
//				return crfoutput;
//			}
//			if (W != GW || H != GH) {
//				printf("Annotation size doesn't match image!\n");
//				return crfoutput;
//			}

			/////////// Put your own unary classifier here! ///////////
			MatrixXf unary = computeUnary(getLabeling(crfmerged.data, W * H, M), M);
			///////////////////////////////////////////////////////////

			// Setup the CRF model
			DenseCRF2D dcrf(W, H, M);
			// Specify the unary potential as an array of size W*H*(#classes)
			// packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
			dcrf.setUnaryEnergy(unary);
			// add a color independent term (feature = pixel location 0..W-1, 0..H-1)
			// x_stddev = 3
			// y_stddev = 3
			// weight = 3
			dcrf.addPairwiseGaussian(3, 3, new PottsCompatibility(3));
			// add a color dependent term (feature = xyrgb)
			// x_stddev = 60
			// y_stddev = 60
			// r_stddev = g_stddev = b_stddev = 20
			// weight = 10
			dcrf.addPairwiseBilateral(50, 50, 5, 5, 5, frame.data,
					new PottsCompatibility(20));

			// Do map inference
			// 	MatrixXf Q = crf.startInference(), t1, t2;
			// 	printf("kl = %f\n", crf.klDivergence(Q) );
			// 	for( int it=0; it<5; it++ ) {
			// 		crf.stepInference( Q, t1, t2 );
			// 		printf("kl = %f\n", crf.klDivergence(Q) );
			// 	}
			// 	VectorXs map = crf.currentMap(Q);
			VectorXs map = dcrf.map(5);
			// Store the result
			unsigned char *res = colorize(map, W, H);
			crfoutput = Mat(H, W, CV_8UC3, res);

//			delete[] im;
//			delete[] anno;
			delete[] res;

			imwrite("crfout.png", crfoutput);
//			imshow("crfoutfromdense", crfoutput);
//			waitKey(0);
			for (int r = 0; r < crfoutput.rows; r++) {
				for (int c = 0; c < crfoutput.cols; c++) {
//					cout<<crfoutput.at<Vec3b>(r,c)<<endl;

					if (crfoutput.at<Vec3b>(r, c) == Vec3b(0, 255, 0)) {
						crfoutput.at<Vec3b>(r, c) = Vec3b(0, 0, 0);
					} else if (crfoutput.at<Vec3b>(r, c) == Vec3b(0, 50, 0)) {
						crfoutput.at<Vec3b>(r, c) = Vec3b(255, 255, 255);
					}
				}
			}
			Mat crfoutputG;
			cvtColor(crfoutput, crfoutputG, CV_BGR2GRAY);
//			imshow("crfsil", crfoutputG);
//			waitKey(0);

			//Closing
			if (domorpho == 1) {
				int morph_size = 1;
				Mat element = getStructuringElement(MORPH_RECT,
						Size(2 * morph_size + 1, 2 * morph_size + 1),
						Point(morph_size, morph_size));
				//			Mat dst;
				for (int i = 1; i < 3; i++) {
					morphologyEx(crfoutputG, crfoutputG, MORPH_CLOSE, element,
							Point(-1, -1), i);
					//morphologyEx( src, dst, MORPH_TOPHAT, element ); // here iteration=1
					//				imshow("source", crfoutputG);
					//				imshow("result", dst);
					//				waitKey(0);
				}
			}

			if (docrf == 1) {
				string outputPath;
				if (countView < 10) {
					outputPath = inputdir + "/Crf/cam0" + to_string(countView)
							+ "/" + to_string(countFrame) + ".png";
				} else if (countView >= 10) {
					outputPath = inputdir + "/Crf/cam" + to_string(countView)
							+ "/" + to_string(countFrame) + ".png";
				}
				imwrite(outputPath, crfoutputG);

//				crfmerged.release();
//				maskcrf.release();
//				maskcrf2.release();
//				crfoutput.release();
//				crf.release();
//				crf2.release();
//				mask.release();
//				mask2.release();
//				objectMask.release();
//				objectMask2.release();
//				fg.release();
//				fg2.release();
//				outDetections.release();
//				outMasks.release();

				return crfoutputG;

			}

		} else {
			Mat mask_fgpf(frame.size(), CV_8UC1, Scalar(0));
			string outputPath;
			if (countView < 10) {
				outputPath = inputdir + "/Crf/cam0" + to_string(countView) + "/"
						+ to_string(countFrame) + ".png";
			} else if (countView >= 10) {
				outputPath = inputdir + "/Crf/cam" + to_string(countView) + "/"
						+ to_string(countFrame) + ".png";
			}
			imwrite(outputPath, mask_fgpf);
			return mask_fgpf;
		}
	}
}
