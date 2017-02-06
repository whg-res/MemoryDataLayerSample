#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <atltime.h>
#include <opencv2/opencv.hpp>

#define WIDTH	224
#define HEIGHT	224
#define CHANNEL	3
#define NCLASS	102

using namespace caffe;
using namespace std;

vector<string> split(string str, char c){
	vector<string> v;
	string buf = "";
	stringstream ss;

	ss << str;
	while (getline(ss, buf, c)){
		v.push_back(buf);
	}

	return v;
}

void readImgListToFloat(string list_path, float *data, float *label, int data_len){

	ifstream ifs;
	string str;
	int n = 0;
	ifs.open(list_path, std::ios::in);
	if (!ifs){ LOG(INFO) << "cannot open " << list_path; return; }

	float mean[CHANNEL] = { 104, 117, 123 };

	while (getline(ifs, str)){
		vector<string> entry = split(str, ' ');
		cout << "reading: " << entry[0] << endl;
		cv::Mat img = cv::imread(entry[0]);
		cv::Mat resized_img;
		cv::resize(img, resized_img, cv::Size(WIDTH, HEIGHT));
		for (int y = 0; y < HEIGHT; y++){
			for (int x = 0; x < WIDTH; x++){
				data[y*resized_img.cols + x + resized_img.cols*resized_img.rows*0 + WIDTH * HEIGHT * CHANNEL * n]
					= resized_img.data[y*resized_img.step + x*resized_img.elemSize() + 0] - mean[0];
				data[y*resized_img.cols + x + resized_img.cols*resized_img.rows*1 + WIDTH * HEIGHT * CHANNEL * n]
					= resized_img.data[y*resized_img.step + x*resized_img.elemSize() + 1] - mean[1];
				data[y*resized_img.cols + x + resized_img.cols*resized_img.rows*2 + WIDTH * HEIGHT * CHANNEL * n]
					= resized_img.data[y*resized_img.step + x*resized_img.elemSize() + 2] - mean[2];
			}
		}
		label[n] = stof(entry[1]);
		n++;
	}
}


void run_googlenet_train(){

	//学習用のデータを取得
	SolverParameter solver_param;
	ReadProtoFromTextFileOrDie("G:\\Projects\\roundRegression\\caffe-master\\caffeTest\\solver.prototxt", &solver_param);
	std::shared_ptr<Solver<float>> solver(SolverRegistry<float>::CreateSolver(solver_param));
	const auto net = solver->net();
	const auto test_net = solver->test_nets();

	//評価用のデータを取得
	net->CopyTrainedLayersFrom("G:\\Projects\\roundRegression\\caffe-master\\caffeTest\\bvlc_googlenet.caffemodel");
	test_net[0]->CopyTrainedLayersFrom("G:\\Projects\\roundRegression\\caffe-master\\caffeTest\\bvlc_googlenet.caffemodel");

	int train_data_size = 2000;
	float *train_input_data;
	float *train_label;

	int test_data_size = 2000;
	float *test_input_data;
	float *test_label;

	//領域確保
	train_input_data	= new float[train_data_size*HEIGHT*WIDTH*CHANNEL];
	train_label			= new float[train_data_size];
	test_input_data		= new float[test_data_size*HEIGHT*WIDTH*CHANNEL];
	test_label			= new float[test_data_size];

	//データ読み込み
	readImgListToFloat("G:\\Projects\\roundRegression\\caffe-master\\caffeTest\\train_r.txt", train_input_data, train_label, train_data_size);
	readImgListToFloat("G:\\Projects\\roundRegression\\caffe-master\\caffeTest\\test_r.txt", test_input_data, test_label, test_data_size);

	//ネットワークに反映
	const auto train_input_layer = boost::dynamic_pointer_cast<MemoryDataLayer<float>>( net->layer_by_name("data") );
	const auto test_input_layer = boost::dynamic_pointer_cast<MemoryDataLayer<float>>( test_net[0]->layer_by_name("data") );
	train_input_layer->Reset((float*)train_input_data, (float*)train_label, train_data_size);
	test_input_layer->Reset((float*)test_input_data, (float*)test_label, test_data_size);

	//学習開始
	LOG(INFO) << "Solve start.";
	solver->Solve();

	//開放
	delete[] train_input_data;	train_input_data = 0;
	delete[] train_label;		train_label = 0;
	delete[] test_input_data;	test_input_data = 0;
	delete[] test_label;		test_label = 0;

}


void run_googlenet_test(){

	int batch_size = 50;
	int test_data_size = 2000;
	int batch_iter = test_data_size / batch_size;
	int test_input_num = 2;

	//データ格納領域宣言
	float *test_input_data;
	float *test_label;
	test_input_data = new float[test_data_size*HEIGHT*WIDTH*CHANNEL];
	test_label = new float[test_data_size];

	//ネットワーク読み込み
	Net<float> test_net("G:\\Projects\\roundRegression\\caffe-master\\caffeTest\\deploy.prototxt", TEST);
	test_net.CopyTrainedLayersFrom("G:\\Projects\\roundRegression\\caffe-master\\caffeTest\\snapshot\\bvlc_googlenet_iter_10000.caffemodel");

	//データ読み込み
	readImgListToFloat("G:\\Projects\\roundRegression\\caffe-master\\caffeTest\\test_r.txt", test_input_data, test_label, test_data_size);

	CFileTime cTimeStart, cTimeEnd;
	CFileTimeSpan cTimeSpan;
	cTimeStart = CFileTime::GetCurrentTime();

	const auto input_test_layer = boost::dynamic_pointer_cast<MemoryDataLayer<float>>(test_net.layer_by_name("data"));
	for (int batch = 0; batch < batch_iter; batch++){
		//入力データを選択的にネットワークにセット＆識別
		input_test_layer->Reset((float*)test_input_data + batch * WIDTH*HEIGHT*CHANNEL * batch_size, (float*)test_label + batch * batch_size, batch_size);
		const auto result = test_net.Forward();

		//結果を受け取り、一番スコアの高いクラスに識別する
		const auto data = result[1]->cpu_data();
		for (int i = 0; i < batch_size; i++){
			int max_id = 0;
			float max = 0;
			for (int j = 0; j < NCLASS; j++){
				if (max < data[i * NCLASS + j]){
					max = data[i * NCLASS + j];
					max_id = j;
				}
			}
			cout << max_id << ", " << max << endl;
		}
	}
	cTimeEnd = CFileTime::GetCurrentTime();
	cTimeSpan = cTimeEnd - cTimeStart;
	cerr << "testing time : " << cTimeSpan.GetTimeSpan() / 10000 << "[ms]" << endl;

	delete[] test_input_data;	test_input_data = 0;
	delete[] test_label;		test_label = 0;

}


int main(int argc, char **argv){
	FLAGS_alsologtostderr = 1;		//コンソールへのログ出力ON
	GlobalInit(&argc, &argv);

	Caffe::set_mode(Caffe::GPU);	//GPUモード

	run_googlenet_train();			//学習
	//run_googlenet_test();			//評価

	return 0;

}