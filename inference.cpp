#include "inference.h"

#include <opencv2/dnn.hpp>
#include <random>
#include <opencv2/imgcodecs.hpp>

#include "timer.h"

namespace yolo
{
	// Constructor to initialize the model with default input shape
	Inference::Inference(const std::string& model_path, const float& model_confidence_threshold,
	                     const float& model_NMS_threshold)
	{
		model_input_shape_ = cv::Size(640, 640);
		// Set the default size for models with dynamic shapes to prevent errors.
		model_confidence_threshold_ = model_confidence_threshold;
		model_NMS_threshold_ = model_NMS_threshold;
		InitializeModel(model_path);
	}

	// Constructor to initialize the model with specified input shape
	Inference::Inference(const std::string& model_path, const cv::Size model_input_shape,
	                     const float& model_confidence_threshold, const float& model_NMS_threshold)
	{
		model_input_shape_ = model_input_shape;
		model_confidence_threshold_ = model_confidence_threshold;
		model_NMS_threshold_ = model_NMS_threshold;
		InitializeModel(model_path);
	}

	void Inference::InitializeModel(const std::string& model_path)
	{
		ov::Core core; // OpenVINO core object
		std::shared_ptr<ov::Model> model = core.read_model(model_path); // Read the model from file

		// If the model has dynamic shapes, reshape it to the specified input shape
		if (model->is_dynamic())
		{
			model->reshape({
				1, 3, static_cast<long int>(model_input_shape_.height), static_cast<long int>(model_input_shape_.width)
			});
		}

		// Preprocessing setup for the model
		auto ppp = ov::preprocess::PrePostProcessor(model);
		ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(
			ov::preprocess::ColorFormat::BGR);
		ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).
		    scale({255, 255, 255});
		ppp.input().model().set_layout("NCHW");
		ppp.output(0).tensor().set_element_type(ov::element::f32);
		ppp.output(0).postprocess().convert_layout({0, 2, 1});
		ppp.output(1).tensor().set_element_type(ov::element::f32);
		model = ppp.build(); // Build the preprocessed model

		// Compile the model for inference
		compiled_model_ = core.compile_model(model, "AUTO");
		inference_request_ = compiled_model_.create_infer_request(); // Create inference request

		// Get input shape from the model
		const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
		const ov::Shape input_shape = inputs[0].get_shape();
		auto height = static_cast<short>(input_shape[1]);
		auto width = static_cast<short>(input_shape[2]);
		model_input_shape_ = cv::Size2f(width, height);

		// Get output shape from the model
		const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
		const ov::Shape output_shape = outputs[0].get_shape();
		height = static_cast<short>(output_shape[1]);
		width = static_cast<short>(output_shape[2]);
		model_output_shape_ = cv::Size(width, height);
	}

	// Method to run inference on an input frame
	void Inference::RunInference(cv::Mat& frame)
	{
		const std::vector<int> classes = {0};

		auto start_time = Timer::get_time();
		const auto preprocess = Preprocessing(frame); // Preprocess the input frame
		Timer::print_time(start_time, "Preprocessing time");

		start_time = Timer::get_time();
		inference_request_.set_input_tensor(preprocess.input_tensor); // Set input tensor for inference
		inference_request_.infer(); // Run inference
		Timer::print_time(start_time, "Inference time");

		const auto output0 = inference_request_.get_output_tensor(0);
		const auto output1 = inference_request_.get_output_tensor(1);

		start_time = Timer::get_time();
		PostProcessing(frame, output0, output1, preprocess.scale_factor, classes); // Postprocess the inference results
		Timer::print_time(start_time, "Postprocessing time");
	}

	// Method to preprocess the input frame
	PreProcessingResult Inference::Preprocessing(const cv::Mat& frame)
	{
		cv::Mat resized_frame;
		int new_width, new_height;
		const int target_width = static_cast<int>(model_input_shape_.width);
		const int target_height = static_cast<int>(model_input_shape_.height);

		if (frame.rows > frame.cols)
		{
			new_height = target_height;
			new_width = static_cast<int>(frame.cols * (static_cast<float>(target_height) / frame.rows));
		}
		else
		{
			new_width = target_width;
			new_height = static_cast<int>(frame.rows * (static_cast<float>(target_width) / frame.cols));
		}
		cv::resize(frame, resized_frame, cv::Size(new_width, new_height));

		// cv::resize(frame, resized_frame, model_input_shape_, 0, 0, cv::INTER_LINEAR);
		constexpr int top = 0;
		const int bottom = target_height - new_height - top;
		constexpr int left = 0;
		const int right = target_width - new_width - left;

		// Resize the frame to match the model input shape
		cv::copyMakeBorder(resized_frame, resized_frame, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

		// Calculate scaling factor
		ScaleFactor scale_factor{};
		scale_factor.x = static_cast<float>(frame.cols) / static_cast<float>(new_width);
		scale_factor.y = static_cast<float>(frame.rows) / static_cast<float>(new_height);

		auto* input_data = reinterpret_cast<float*>(resized_frame.data); // Get pointer to resized frame data
		const auto input_tensor = ov::Tensor(compiled_model_.input().get_element_type(),
		                                     compiled_model_.input().get_shape(), input_data); // Create input tensor

		return {scale_factor, input_tensor};
	}

	// Method to postprocess the inference results
	void Inference::PostProcessing(cv::Mat& frame, const ov::Tensor& output0, const ov::Tensor& output1,
	                               const ScaleFactor& scale_factor, const std::vector<int>& classes)
	{
		std::vector<int> class_list;
		std::vector<float> confidence_list;
		std::vector<cv::Rect> boxes;
		std::vector<cv::Mat> masks;

		const auto& shape = output0.get_shape();
		const auto& proto_shape = output1.get_shape();
		// Get the output tensor from the inference request
		const cv::Mat detection_outputs(static_cast<int>(shape[1]), static_cast<int>(shape[2]), CV_32F,
		                                output0.data<float>()); // Create OpenCV matrix from output tensor
		const cv::Mat proto(static_cast<int>(proto_shape[1]), static_cast<int>(proto_shape[2] * proto_shape[3]), CV_32F,
		                    output1.data<float>()); //[32,25600]

		const auto class_size = static_cast<int>(classes_.size());
		const auto end_col = static_cast<int>(shape[2]);
		// Iterate over detections and collect class IDs, confidence scores, and bounding boxes
		for (int i = 0; i < detection_outputs.rows; ++i)
		{
			const cv::Mat classes_scores = detection_outputs.row(i).colRange(4, class_size + 4);

			cv::Point class_id;
			double score;
			cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id); // Find the class with the highest score

			// Check if the detection meets the confidence threshold
			if (score > model_confidence_threshold_)
			{
				class_list.push_back(class_id.y);
				confidence_list.push_back(static_cast<float>(score));

				const float x = detection_outputs.at<float>(i, 0);
				const float y = detection_outputs.at<float>(i, 1);
				const float w = detection_outputs.at<float>(i, 2);
				const float h = detection_outputs.at<float>(i, 3);

				cv::Rect box;
				box.x = static_cast<int>(x);
				box.y = static_cast<int>(y);
				box.width = static_cast<int>(w);
				box.height = static_cast<int>(h);
				boxes.push_back(box);

				cv::Mat mask = detection_outputs.row(i).colRange(class_size + 4, end_col);
				masks.push_back(mask);
			}
		}

		// Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes
		std::vector<int> NMS_result;
		cv::dnn::NMSBoxes(boxes, confidence_list, model_confidence_threshold_, model_NMS_threshold_, NMS_result);

		cv::Mat rgb_mask = cv::Mat::zeros(frame.size(), frame.type());

		// Collect final detections after NMS
		for (const unsigned int id : NMS_result)
		{
			const auto class_id = static_cast<short>(class_list[id]);
			bool skip = true;
			for (auto class_ : classes)
			{
				if (class_id == class_)
				{
					skip = false;
					break;
				}
			}

			if (skip)
			{
				continue;
			}

			const auto box = GetBoundingBox(boxes[id], scale_factor);

			// Detection result{};
			// result.class_id = class_list[id];
			// result.confidence = confidence_list[id];
			// result.x = box.x;
			// result.y = box.y;
			// result.w = box.width;
			// result.h = box.height;

			// DrawDetectedObject(frame, result);
			ApplyMask(frame, masks[id], proto, box, rgb_mask, scale_factor);
		}

		cv::Mat roi;
		cv::GaussianBlur(frame & rgb_mask, roi, cv::Size(0, 0), 23, 23);
		frame = (frame & (~rgb_mask)) + roi;
	}

	void Inference::ApplyMask(const cv::Mat& frame, const cv::Mat& detectedMask, const cv::Mat& proto,
	                          const cv::Rect& box, const cv::Mat& rgb_mask, const ScaleFactor& scale_factor)
	{
		cv::Mat m = detectedMask * proto;
		for (int col = 0; col < m.cols; col++)
		{
			m.at<float>(0, col) = sigmoid_function(m.at<float>(0, col));
		}

		auto m1 = m.reshape(1, 160); // 1x25600 -> 160x160
		int x1 = std::max(0, box.x);
		int y1 = std::max(0, box.y);
		int x2 = std::max(0, box.br().x);
		int y2 = std::max(0, box.br().y);
		int mx1 = static_cast<int>(static_cast<float>(x1) / scale_factor.x * 0.25);
		int my1 = static_cast<int>(static_cast<float>(y1) / scale_factor.y * 0.25);
		int mx2 = static_cast<int>(static_cast<float>(x2) / scale_factor.x * 0.25);
		int my2 = static_cast<int>(static_cast<float>(y2) / scale_factor.y * 0.25);

		cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2));
		cv::Mat rm, det_mask;
		cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));

		for (int r = 0; r < rm.rows; r++)
		{
			for (int c = 0; c < rm.cols; c++)
			{
				float pv = rm.at<float>(r, c);
				if (pv > 0.5)
				{
					rm.at<float>(r, c) = 1.0;
				}
				else
				{
					rm.at<float>(r, c) = 0.0;
				}
			}
		}
		rm = rm * 255;
		rm.convertTo(det_mask, CV_8UC1);
		if ((y1 + det_mask.rows) >= frame.rows)
		{
			y2 = frame.rows - 1;
		}
		if ((x1 + det_mask.cols) >= frame.cols)
		{
			x2 = frame.cols - 1;
		}

		cv::Mat mask = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);
		det_mask(cv::Range(0, y2 - y1), cv::Range(0, x2 - x1)).copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2)));
		add(rgb_mask, cv::Scalar(255, 255, 255), rgb_mask, mask);
	}

	void Inference::WriteImage(const cv::Mat& img, const std::string& image_name)
	{
		image_name_index++;
		const std::string full_image_name = "debug/" + std::to_string(image_name_index) + "_image_" + image_name +
			".jpg";
		cv::imwrite(full_image_name, img);
	}

	float Inference::sigmoid_function(const float a)
	{
		const float b = 1.f / (1.f + exp(-a));
		return b;
	}

	// Method to get the bounding box in the correct scale
	cv::Rect Inference::GetBoundingBox(const cv::Rect& src, const ScaleFactor& scale_factor)
	{
		cv::Rect box = src;
		box.x = (box.x - box.width / 2) * scale_factor.x;
		box.y = (box.y - box.height / 2) * scale_factor.y;
		box.width *= scale_factor.x;
		box.height *= scale_factor.y;
		return box;
	}

	void Inference::DrawDetectedObject(cv::Mat& frame, const Detection& detection) const
	{
		const float& confidence = detection.confidence;
		const int& class_id = detection.class_id;

		// Generate a random color for the bounding box
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dis(120, 255);
		const auto& color = cv::Scalar(dis(gen), dis(gen), dis(gen));

		// Draw the bounding box around the detected object
		cv::rectangle(frame, cv::Point(detection.x, detection.y),
		              cv::Point(detection.x + detection.w, detection.y + detection.h), color, 3);

		// Prepare the class label and confidence text
		const std::string class_string = classes_[class_id] + std::to_string(confidence).substr(0, 4);

		// Get the size of the text box
		const cv::Size textSize = cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 0.75, 2, nullptr);
		const cv::Rect textBox(detection.x, detection.y - 40, textSize.width + 10, textSize.height + 20);

		// Draw the text box
		cv::rectangle(frame, textBox, color, cv::FILLED);

		// Put the class label and confidence text above the bounding box
		cv::putText(frame, class_string, cv::Point(detection.x + 5, detection.y - 10), cv::FONT_HERSHEY_DUPLEX, 0.75,
		            cv::Scalar(0, 0, 0), 2, 0);
	}
} // namespace yolo
