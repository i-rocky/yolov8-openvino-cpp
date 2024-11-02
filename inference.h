#ifndef YOLO_INFERENCE_H_
#define YOLO_INFERENCE_H_

#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

namespace yolo
{
	struct Detection
	{
		short class_id;
		float confidence;
		int x;
		int y;
		int w;
		int h;
	};

	struct ScaleFactor
	{
		float x;
		float y;
	};

	struct PreProcessingResult
	{
		ScaleFactor scale_factor;
		ov::Tensor input_tensor;
	};

	class Inference
	{
	public:
		Inference() = default;
		// Constructor to initialize the model with default input shape
		Inference(const std::string& model_path, const float& model_confidence_threshold,
		          const float& model_NMS_threshold);
		// Constructor to initialize the model with specified input shape
		Inference(const std::string& model_path, cv::Size model_input_shape, const float& model_confidence_threshold,
		          const float& model_NMS_threshold);

		void RunInference(cv::Mat& frame);

	private:
		void InitializeModel(const std::string& model_path);
		PreProcessingResult Preprocessing(const cv::Mat& frame);
		void PostProcessing(cv::Mat& frame, const ov::Tensor& output0, const ov::Tensor& output1,
		                    const ScaleFactor& scale_factor, const std::vector<int>& classes);
		static cv::Rect GetBoundingBox(const cv::Rect& src, const ScaleFactor& scale_factor);
		void DrawDetectedObject(cv::Mat& frame, const Detection& detection) const;
		static float sigmoid_function(float a);
		void ApplyMask(const cv::Mat& frame, const cv::Mat& detectedMask, const cv::Mat& proto, const cv::Rect& box, const cv::Mat& rgb_mask, const ScaleFactor& scale_factor);
		void WriteImage(const cv::Mat& img, const std::string& image_name);

		int image_name_index = 0;
		cv::Point2f scale_factor_; // Scaling factor for the input frame
		cv::Size2f model_input_shape_; // Input shape of the model
		cv::Size model_output_shape_; // Output shape of the model

		ov::InferRequest inference_request_; // OpenVINO inference request
		ov::CompiledModel compiled_model_; // OpenVINO compiled model

		float model_confidence_threshold_{}; // Confidence threshold for detections
		float model_NMS_threshold_{}; // Non-Maximum Suppression threshold

		std::vector<std::string> classes_{
			"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
			"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
			"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
			"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
			"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
			"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
			"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
			"cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
			"scissors", "teddy bear", "hair drier", "toothbrush"
		};
	};
} // namespace yolo

#endif // YOLO_INFERENCE_H_
