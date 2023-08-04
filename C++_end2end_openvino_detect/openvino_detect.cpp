#include "openvino_detect.h"

using namespace std;
using namespace cv;

yolo_detct::yolo_detct() {
    model = core.read_model(MODEL_PATH);
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    // Specify input image format
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(
            ov::preprocess::ColorFormat::BGR);
    // Specify preprocess pipeline to input image without resizing
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(
            ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});
    //  Specify model's input layout
    ppp.input().model().set_layout("NCHW");
    // Specify output results format
    ppp.output().tensor().set_element_type(ov::element::f32);
    // Embed above steps in the graph
    model = ppp.build();
    compiled_model = core.compile_model(model, DEVICE);
}


yolo_detct::Resize resize_and_pad(Mat &img, Size new_shape) {
    float width = img.cols;
    float height = img.rows;
    float r = float(new_shape.width / max(width, height));
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    yolo_detct::Resize resize_pic;
    resize(img, resize_pic.resized_image, Size(new_unpadW, new_unpadH), 0, 0, INTER_AREA);

    resize_pic.dw = new_shape.width - new_unpadW;
    resize_pic.dh = new_shape.height - new_unpadH;
    Scalar color = Scalar(100, 100, 100);
    copyMakeBorder(resize_pic.resized_image, resize_pic.resized_image, 0, resize_pic.dh, 0, resize_pic.dw, BORDER_CONSTANT,
                   color);

    return resize_pic;
}

std::vector<yolo_detct::Object> yolo_detct::work(Mat img) {
    Resize res = resize_and_pad(img, Size(IMG_SIZE, IMG_SIZE));
    // Step 5. Create tensor from image
    float *input_data = (float *) res.resized_image.data;
    input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(),
                                         input_data);
    // Step 6. Create an infer request for model inference
    infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    //Step 7. Retrieve inference results
    const ov::Tensor &output_tensor = infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape();
    float *detections = output_tensor.data<float>();
    // Step 8. Postprocessing including NMS
    std::vector<Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;

    for (int i = 0; i < output_shape[1]; i++) {
        float *detection = &detections[i * output_shape[2]];

        float confidence = detection[4];
        if (confidence >= CONF_THRESHOLD) {
            float *classes_scores = &detection[5];
            Mat scores(1, output_shape[2] - 5, CV_32FC1, classes_scores);
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            if (max_class_score > SCORE_THRESHOLD) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                float x = detection[0];
                float y = detection[1];
                float w = detection[2];
                float h = detection[3];
                float xmin = x - (w / 2);
                float ymin = y - (h / 2);
                boxes.push_back(Rect(xmin, ymin, w, h));
            }
        }
    }
    std::vector<int> nms_result;
    dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    std::vector<yolo_detct::Object> output;
    for (int i = 0; i < nms_result.size(); i++) {
        yolo_detct::Object result;
        int idx = nms_result[i];
        result.label = class_ids[idx];
        result.prob = confidences[idx];
        result.rect = boxes[idx];
        output.push_back(result);
    }

#ifdef VIDEO
    for (int i = 0; i < output.size(); i++) {
        auto detection = output[i];
        auto box = detection.rect;
        float rx = (float) img.cols / (float) (res.resized_image.cols - res.dw);
        float ry = (float) img.rows / (float) (res.resized_image.rows - res.dh);
        box.x = rx * box.x;
        box.y = ry * box.y;
        box.width = rx * box.width;
        box.height = ry * box.height;
        float xmax = box.x + box.width;
        float ymax = box.y + box.height;
        rectangle(img, Point(box.x, box.y), Point(xmax, ymax), Scalar(0, 255, 0), 1);
        rectangle(img, Point(box.x, box.y - 10), Point(xmax, box.y), Scalar(0, 255, 0), FILLED);
        putText(img, class_names[detection.label], Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(0, 0, 0));
    }
    imshow("detect", img);
    waitKey(1);
#endif
    return output;
}