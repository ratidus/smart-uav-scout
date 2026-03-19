#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h> // The core ONNX Runtime API
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

// ==========================================
// --- SMART CENTROID + IOU TRACKER ---
// ==========================================

// Structure to hold data for a tracked object over time
struct Track {
    int id;               
    int class_id;         
    cv::Rect box;         
    int frames_lost;      
    cv::Point2f center;   // NEW: We now remember the exact center of the object
};

class SmartTracker {
private:
    int next_id = 1;                
    // 1. FIX FLICKERING: Increase patience to 30 frames (1 second at 30fps)
    int max_lost_frames = 30;       
    float iou_threshold = 0.2f;     // Lowered threshold to catch fast-moving objects

    // 1. FIXED: Explicitly cast the integer area to float to satisfy the compiler
    static float calculateIoU(const cv::Rect& a, const cv::Rect& b) {
        float intersection_area = static_cast<float>((a & b).area());
        float union_area = static_cast<float>(a.area() + b.area()) - intersection_area;
        return union_area > 0.0f ? intersection_area / union_area : 0.0f;
    }

    // 2. FIXED & OPTIMIZED: Avoid slow std::pow, use raw float multiplication
    static float calculateDistance(const cv::Point2f& a, const cv::Point2f& b) {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        // In C++, std::sqrt applied to floats returns a float (no data loss warning)
        return std::sqrt(dx * dx + dy * dy); 
    }

    // NEW: Get the center point of a bounding box
    static cv::Point2f getCenter(const cv::Rect& box) {
        return cv::Point2f(box.x + box.width / 2.0f, box.y + box.height / 2.0f);
    }

public:
    std::vector<Track> tracks; 

    std::vector<Track> update(const std::vector<Detection>& detections) {
        std::vector<Track> updated_tracks;
        std::vector<bool> det_matched(detections.size(), false);
        std::vector<bool> trk_matched(tracks.size(), false);

        // 1. MATCHING: Combine IoU and Distance for robust tracking
        for (size_t d = 0; d < detections.size(); ++d) {
            int best_track_idx = -1;
            float best_iou = iou_threshold; 
            float min_distance = 10000.0f; // Start with a huge distance
            cv::Point2f det_center = getCenter(detections[d].box);

            for (size_t t = 0; t < tracks.size(); ++t) {
                if (trk_matched[t] || tracks[t].class_id != detections[d].class_id) continue;
                
                float iou = calculateIoU(detections[d].box, tracks[t].box);
                float distance = calculateDistance(det_center, tracks[t].center);

                // FIX INTERSECTION: If IoU is acceptable, pick the track that is physically closest to the old center
                if (iou >= iou_threshold && distance < min_distance) {
                    best_iou = iou;
                    min_distance = distance;
                    best_track_idx = static_cast<int>(t);
                }
            }

            if (best_track_idx >= 0) {
                trk_matched[best_track_idx] = true;
                det_matched[d] = true;
                tracks[best_track_idx].box = detections[d].box;
                tracks[best_track_idx].center = det_center; // Update center
                tracks[best_track_idx].frames_lost = 0;
                updated_tracks.push_back(tracks[best_track_idx]);
            }
        }

        // 2. BIRTH: New objects
        for (size_t d = 0; d < detections.size(); ++d) {
            if (!det_matched[d]) {
                updated_tracks.push_back({next_id++, detections[d].class_id, detections[d].box, 0, getCenter(detections[d].box)});
            }
        }

        // 3. GRACE PERIOD: Keep occluded objects in memory longer
        for (size_t t = 0; t < tracks.size(); ++t) {
            if (!trk_matched[t]) {
                tracks[t].frames_lost++;
                if (tracks[t].frames_lost < max_lost_frames) {
                    updated_tracks.push_back(tracks[t]);
                }
            }
        }

        tracks = updated_tracks;
        return tracks;
    }
};

// ==========================================
// --- PRE-PROCESSING FUNCTION (LETTERBOXING) ---
// Formats the image to a 640x640 square without distorting the aspect ratio.
// Returns a flat vector of floats in NCHW format (Batch, Channels, Height, Width).
std::vector<float> format_yolov8(const cv::Mat& source, cv::Mat& out_letterbox) {
    
    // 1. Calculate the scaling factor
    float width = static_cast<float>(source.cols);
    float height = static_cast<float>(source.rows);
    float scale = std::min(INPUT_WIDTH / width, INPUT_HEIGHT / height);
    
    // Calculate the new dimensions of the image after scaling
    int new_unpad_w = static_cast<int>(std::round(width * scale));
    int new_unpad_h = static_cast<int>(std::round(height * scale));

    // 2. Resize the image while maintaining its aspect ratio
    cv::Mat resized_img;
    cv::resize(source, resized_img, cv::Size(new_unpad_w, new_unpad_h));

    // 3. Create the gray canvas (padding)
    int dw = static_cast<int>(INPUT_WIDTH) - new_unpad_w;
    int dh = static_cast<int>(INPUT_HEIGHT) - new_unpad_h;

    dw /= 2;
    dh /= 2;

    cv::copyMakeBorder(resized_img, out_letterbox, dh, static_cast<int>(INPUT_HEIGHT) - new_unpad_h - dh, 
                       dw, static_cast<int>(INPUT_WIDTH) - new_unpad_w - dw, 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // 4. Memory Repacking (HWC to NCHW) and Normalization (/255.0)
    cv::Mat blob;
    cv::dnn::blobFromImage(out_letterbox, blob, 1.0 / 255.0, cv::Size(static_cast<int>(INPUT_WIDTH), static_cast<int>(INPUT_HEIGHT)), cv::Scalar(), true, false);

    // 5. Convert the multidimensional cv::Mat blob into a flat 1D std::vector<float>
    size_t blob_size = blob.total(); // Total number of elements (1 * 3 * 640 * 640)
    std::vector<float> input_tensor_values(blob_size);
    std::memcpy(input_tensor_values.data(), blob.ptr<float>(), blob_size * sizeof(float));

    return input_tensor_values;
}

// --- POST-PROCESSING FUNCTION ---
// Converts the raw tensor output from YOLOv8 into usable bounding boxes
// Reverses the letterboxing effect to map coordinates back to the original video resolution
std::vector<Detection> postprocess(cv::Mat& input_image, std::vector<cv::Mat>& outputs, const std::vector<std::string>& class_names) {
    std::vector<Detection> detections;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // 1. Tensor Parsing
    cv::Mat data = outputs[0];
    cv::Mat rows = data.reshape(1, 14); // Reshape to [14, 8400]
    cv::transpose(rows, rows);          // Transpose to [8400, 14]

    float* data_ptr = (float*)rows.data;
    int rows_count = rows.rows;       
    int dimensions = rows.cols;       
    int num_classes = static_cast<int>(class_names.size());

    // --- REVERSE LETTERBOXING MATH ---
    // We calculate the exact same scale and padding that were used in pre-processing
    float img_width = static_cast<float>(input_image.cols);
    float img_height = static_cast<float>(input_image.rows);
    float scale = std::min(INPUT_WIDTH / img_width, INPUT_HEIGHT / img_height);
    
    // Calculate the padding that was added to the image
    float pad_w = (INPUT_WIDTH - img_width * scale) / 2.0f;
    float pad_h = (INPUT_HEIGHT - img_height * scale) / 2.0f;

    // 2. Iterate through all 8400 anchors
    for (int i = 0; i < rows_count; ++i) {
        float* row_ptr = data_ptr + (i * dimensions);
        float* classes_scores = row_ptr + 4; 
        
        cv::Mat scores(1, num_classes, CV_32FC1, classes_scores);
        cv::Point class_id_point;
        double max_class_score;
        minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

        // 3. Filter out weak predictions early to save CPU time
        if (max_class_score > SCORE_THRESHOLD) {
            
            // Raw YOLO coordinates relative to the padded 640x640 square
            float x = row_ptr[0];
            float y = row_ptr[1];
            float w = row_ptr[2];
            float h = row_ptr[3];

            // REMOVE PADDING AND RESCALE: Map back to the original 1920x1080 resolution
            x = (x - pad_w) / scale;
            y = (y - pad_h) / scale;
            w = w / scale;
            h = h / scale;

            // Convert to top-left corner coordinates for cv::Rect
            int left = static_cast<int>(x - 0.5f * w);
            int top = static_cast<int>(y - 0.5f * h);
            int width_box = static_cast<int>(w);
            int height_box = static_cast<int>(h);

            class_ids.push_back(class_id_point.x);
            confidences.push_back(static_cast<float>(max_class_score));
            boxes.push_back(cv::Rect(left, top, width_box, height_box));
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

    // ==========================================
    // 1. ONNX RUNTIME INITIALIZATION
    // ==========================================
    std::cout << "[INFO] Initializing ONNX Runtime..." << std::endl;

    // Create the ORT Environment (handles logging and internal threads)
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SmartUAVScout");

    // Configure session options (how the model should run)
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    
    // Graph optimizations make the model run faster by fusing layers
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // --- GRACEFUL DEGRADATION: Try CUDA, fallback to CPU ---
    try {
        std::cout << "[INFO] Attempting to enable CUDA Execution Provider for RTX 3050..." << std::endl;
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0; // Use the first GPU (your RTX 3050)
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "[SUCCESS] CUDA Provider appended successfully!" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "[WARNING] CUDA failed or not found. Falling back to CPU. Error: " << e.what() << std::endl;
    }

    // Load the model into the session
    // Note: Windows ORT API requires wide strings (std::wstring) for file paths
    std::cout << "[INFO] Loading ONNX model from: " << model_path << std::endl;
    std::wstring widestr(model_path.begin(), model_path.end());
    
    // We use a pointer here so we can initialize it inside a try-catch block later if needed,
    // but for now, we'll initialize it directly.
    Ort::Session* session = nullptr;
    try {
        session = new Ort::Session(env, widestr.c_str(), session_options);
        std::cout << "[SUCCESS] Model loaded into ONNX Runtime Session inside VRAM!" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] Failed to load ONNX model: " << e.what() << std::endl;
        return -1;
    }

    // ==========================================
    // 1.5 INITIALIZE VIDEO CAPTURE
    // ==========================================
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

    // ==========================================
    // 2. MAIN INFERENCE LOOP (ONNX RUNTIME)
    // ==========================================
    cv::Mat frame;
    
    // We need to tell ONNX Runtime the names of the input and output nodes of our YOLOv8 model
    std::vector<const char*> input_names = {"images"};
    std::vector<const char*> output_names = {"output0"};

    // Initialize the Object Tracker
    SmartTracker tracker;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "[INFO] End of video stream reached." << std::endl;
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // --- PRE-PROCESSING ---
        cv::Mat letterbox_img;
        // Our new custom function converts the frame to a flat NCHW vector
        std::vector<float> input_tensor_values = format_yolov8(frame, letterbox_img);

        // Define the shape of our input tensor: [Batch=1, Channels=3, Height=640, Width=640]
        std::vector<int64_t> input_shape = {1, 3, static_cast<int64_t>(INPUT_HEIGHT), static_cast<int64_t>(INPUT_WIDTH)};

        // Create the memory bridge (Tensor) between C++ and ONNX Runtime
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            input_tensor_values.data(), 
            input_tensor_values.size(), 
            input_shape.data(), 
            input_shape.size()
        );

        // --- FORWARD PASS (INFERENCE ON RTX 3050) ---
        // session->Run is where the GPU actually crunches the numbers
        std::vector<Ort::Value> output_tensors = session->Run(
            Ort::RunOptions{nullptr}, 
            input_names.data(), 
            &input_tensor, 
            1, 
            output_names.data(), 
            1
        );

        // --- POST-PROCESSING ---
        // Extract the raw float pointer from the output tensor
        float* raw_output = output_tensors[0].GetTensorMutableData<float>();
        
        // Wrap the raw pointer in a cv::Mat so our existing postprocess function can read it
        // YOLOv8 shape is [1, 14, 8400]
        std::vector<int> sizes = {1, 14, 8400};
        cv::Mat output_mat(sizes, CV_32F, raw_output);
        std::vector<cv::Mat> outputs_vector = {output_mat};

        // Pass the data to our existing filtering function
        std::vector<Detection> results = postprocess(frame, outputs_vector, CLASS_NAMES);

        // --- TRACKING MAGIC ---
        // Feed the raw YOLO detections into our memory module
        std::vector<Track> tracked_objects = tracker.update(results);

        // --- DRAWING TRACKED RESULTS ---
        for (const auto& track : tracked_objects) {
            // Draw a yellow box if the object is currently "lost" (occluded), otherwise green
            cv::Scalar color = (track.frames_lost > 0) ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0); 
            
            cv::rectangle(frame, track.box, color, 2);

            // Create a label with the unique ID
            std::string label = "[ID:" + std::to_string(track.id) + "] " + CLASS_NAMES[track.class_id];
            
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(frame, cv::Point(track.box.x, track.box.y - labelSize.height),
                          cv::Point(track.box.x + labelSize.width, track.box.y + baseLine), color, cv::FILLED);
            
            cv::putText(frame, label, cv::Point(track.box.x, track.box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        // Calculate and display FPS
        auto end = std::chrono::high_resolution_clock::now();
        float fps = 1000.0f / static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        cv::putText(frame, "FPS: " + std::to_string(fps).substr(0, 4), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Smart UAV Scout Inference (ONNX Runtime)", frame);

        if (cv::waitKey(1) == 27) break; 
    }

    delete session;
    cap.release();
    cv::destroyAllWindows();

    return 0;
    
}