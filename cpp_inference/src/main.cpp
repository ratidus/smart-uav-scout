#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>

// --- CONFIGURATION ---
// YOLOv8 default input size
const float INPUT_WIDTH = 640.0f;
const float INPUT_HEIGHT = 640.0f;

// Thresholds for filtering detections
const float SCORE_THRESHOLD = 0.5f; // Minimum confidence to keep a detection (50%)
const float NMS_THRESHOLD = 0.45f;  // Threshold for Non-Maximum Suppression (Overlap)

// VisDrone Classes (Order must exactly match the YAML file used during training!)
const std::vector<std::string> CLASS_NAMES = {
    "Pedestrian", "People", "Bicycle", "Car", "Van",
    "Truck", "Tricycle", "Awning-tricycle", "Bus", "Motor"
};

// Structure to hold a single detected object's data
struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

// --- POST-PROCESSING FUNCTION ---
// Converts the raw tensor output from YOLOv8 into usable bounding boxes
std::vector<Detection> postprocess(cv::Mat& input_image, std::vector<cv::Mat>& outputs, const std::vector<std::string>& class_names) {
    std::vector<Detection> detections;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // 1. Tensor Parsing
    // YOLOv8 output shape: [1, 14, 8400] -> (batch, attributes, anchors)
    // Attributes (14): x, y, w, h, class_0_prob ... class_9_prob
    cv::Mat data = outputs[0];
    cv::Mat rows = data.reshape(1, 14); // Reshape to [14, 8400]
    cv::transpose(rows, rows);          // Transpose to [8400, 14] for efficient memory access

    // Get a raw pointer to the float data for maximum speed
    float* data_ptr = (float*)rows.data;

    int rows_count = rows.rows;       // 8400 possible bounding boxes
    int dimensions = rows.cols;       // 14 elements per box
    int num_classes = static_cast<int>(class_names.size()); // Cast to int to prevent C4267 warning

    // Calculate scaling factors to resize bounding boxes back to the original video resolution
    float x_factor = static_cast<float>(input_image.cols) / INPUT_WIDTH;
    float y_factor = static_cast<float>(input_image.rows) / INPUT_HEIGHT;

    // 2. Iterate through all 8400 anchors
    for (int i = 0; i < rows_count; ++i) {
        // Pointer to the start of the current row (anchor)
        float* row_ptr = data_ptr + (i * dimensions);

        // Pointer to the start of the class scores (skip the first 4 elements: x, y, w, h)
        float* classes_scores = row_ptr + 4; 
        
        // Create a 1D OpenCV matrix to easily find the maximum score among all classes
        cv::Mat scores(1, num_classes, CV_32FC1, classes_scores);
        cv::Point class_id_point;
        double max_class_score;
        
        // Find the index (class_id) and value (max_class_score) of the highest probability
        minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

        // 3. Filter out weak predictions early to save CPU time
        if (max_class_score > SCORE_THRESHOLD) {
            
            // YOLO output coordinates are center X, center Y, width, and height
            float x = row_ptr[0];
            float y = row_ptr[1];
            float w = row_ptr[2];
            float h = row_ptr[3];

            // Convert to top-left corner coordinates and scale to original image size
            int left = static_cast<int>((x - 0.5f * w) * x_factor);
            int top = static_cast<int>((y - 0.5f * h) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            // Store results for NMS processing
            class_ids.push_back(class_id_point.x);
            confidences.push_back(static_cast<float>(max_class_score));
            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    // 4. Non-Maximum Suppression (NMS)
    // Removes overlapping bounding boxes for the same object
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    // 5. Build the final list of valid detections
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        Detection det;
        det.class_id = class_ids[idx];
        det.confidence = confidences[idx];
        det.box = boxes[idx];
        detections.push_back(det);
    }

    return detections;
}

// --- MAIN INFERENCE LOOP ---
int main(int argc, char** argv) {
    
    // Default fallback values
    std::string model_path = "smart_scout_v1.onnx";
    std::string video_source = "video.mp4"; // Changed to process the local video file by default

    // Override defaults if the user provides arguments in the terminal
    if (argc >= 2) {
        model_path = argv[1]; 
    }
    if (argc >= 3) {
        video_source = argv[2];
    }

    std::cout << "[INFO] Loading model from: " << model_path << std::endl;
    std::cout << "[INFO] Video source: " << video_source << std::endl;

    // Load the neural network
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(model_path);
    } catch (cv::Exception& e) {
        std::cerr << "[ERROR] Could not load model. Ensure the path is correct." << std::endl;
        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
        return -1;
    }

    // Use CPU backend for baseline inference
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Initialize video capture
    cv::VideoCapture cap;
    if (video_source == "0") {
        cap.open(0); 
    } else {
        cap.open(video_source); 
    }

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Could not open video source: " << video_source << std::endl;
        std::cerr << "Hint: Ensure 'video.mp4' is in the exact same directory as the executable." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "[INFO] End of video stream reached." << std::endl;
            break;
        }

        // Start timer for FPS calculation
        auto start = std::chrono::high_resolution_clock::now();

        // --- PRE-PROCESSING ---
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(static_cast<int>(INPUT_WIDTH), static_cast<int>(INPUT_HEIGHT)), cv::Scalar(), true, false);
        net.setInput(blob);

        // --- FORWARD PASS (INFERENCE) ---
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // --- POST-PROCESSING ---
        std::vector<Detection> results = postprocess(frame, outputs, CLASS_NAMES);

        // --- DRAWING RESULTS ---
        for (const auto& det : results) {
            cv::Scalar color(0, 255, 0); // Green bounding box
            
            cv::rectangle(frame, det.box, color, 2);

            std::string label = CLASS_NAMES[det.class_id] + ": " + std::to_string(det.confidence).substr(0, 4);
            
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(frame, cv::Point(det.box.x, det.box.y - labelSize.height),
                          cv::Point(det.box.x + labelSize.width, det.box.y + baseLine), color, cv::FILLED);
            
            cv::putText(frame, label, cv::Point(det.box.x, det.box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        // Calculate and display FPS
        auto end = std::chrono::high_resolution_clock::now();
        float fps = 1000.0f / static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        cv::putText(frame, "FPS: " + std::to_string(fps).substr(0, 4), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

        // Show the output frame
        cv::imshow("Smart UAV Scout Inference", frame);

        // Break the loop if the 'ESC' key is pressed
        if (cv::waitKey(1) == 27) break; 
    }

    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}