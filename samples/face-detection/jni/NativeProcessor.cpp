#include <DetectionBasedTracker_jni.h>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include <string>
#include <vector>

#include <android/log.h>

//gwas
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include "cobValue.h"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/videoio.hpp>


#define LOG_TAG "FaceDetection/DetectionBasedTracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

using namespace std;
using namespace cv;
using namespace dnn;

inline void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& mat)
{
    mat = Mat(v_rect, true);
}

///////////////// native call handler ////////////////////

static long handler_convertToGray(cob::ValueMap& params)
{
    long srcAddr = params["src"].asLong();
    long dstAddr = params["dst"].asLong();
    if (srcAddr != 0 && dstAddr != 0)
    {
        Mat &srcMat = *((Mat *) srcAddr);
        Mat &dstMat = *((Mat *) dstAddr);
        cvtColor(srcMat, dstMat, COLOR_RGB2GRAY);
    }
    return 0;
}

static long handler_drawRect(cob::ValueMap& params)
{
    long srcAddr = params["src"].asLong();
    long dstAddr = params["dst"].asLong();
    if (srcAddr != 0 && dstAddr != 0)
    {
        Mat &srcMat = *((Mat *) srcAddr);
        Mat &dstMat = *((Mat *) dstAddr);
        Mat tempMat;

        //cvtColor(srcMat, tempMat, COLOR_RGB2GRAY);

        Point pt1(100, 900);
        Point pt2(800, 100);
        dstMat = srcMat;
        rectangle(dstMat, pt1, pt2, Scalar(255, 0, 0, 255), 3);
    }
    return 0;
}

static long handler_detectBlob(cob::ValueMap& params)
{
    long srcAddr = params["src"].asLong();
    long dstAddr = params["dst"].asLong();
    if (srcAddr != 0 && dstAddr != 0)
    {
        Mat &srcMat = *((Mat *) srcAddr);
        Mat &dstMat = *((Mat *) dstAddr);
        Mat tempMat;

        cvtColor(srcMat, tempMat, COLOR_RGB2GRAY);

        // Setup SimpleBlobDetector parameters.
        SimpleBlobDetector::Params params;

        // Change thresholds
        params.minThreshold = 10;
        params.maxThreshold = 200;

        // Filter by Area.
        params.filterByArea = true;
        params.minArea = 1500;

        // Filter by Circularity
        params.filterByCircularity = true;
        params.minCircularity = 0.1;

        // Filter by Convexity
        params.filterByConvexity = true;
        params.minConvexity = 0.87;

        // Filter by Inertia
        params.filterByInertia = true;
        params.minInertiaRatio = 0.01;

        // Set up the detector with default parameters.
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

        // Detect blobs
        std::vector<KeyPoint> keypoints;
        detector->detect(tempMat, keypoints);

        // Draw detected blobs as red circles.
        // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
        drawKeypoints(tempMat, keypoints, dstMat, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    }
    return 0;
}

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;
Net predictNet;

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

static void splitByLine(std::vector<char>& content, vector<string>& ret)
{
    int count = 0;
    auto itStart = content.begin();
    for (int i = 0; i < content.size(); i++)
    {
        char c = *(content.begin() + i);
        if (c == '\n')
        {
            if (count > 0)
            {
                std::string str;
                str.insert(str.begin(), itStart, itStart + count);
                ret.push_back(str);
            }
            itStart += count + 1;
            count = 0;
        }
        else
        {
            count++;
        }
    }

    if (count > 0)
    {
        std::string str;
        str.insert(str.begin(), itStart, itStart + count);
        ret.push_back(str);
    }
}


static long handler_configNet(cob::ValueMap& params)
{
    auto cfgAddr = params["cfg"].asLong();
    auto modelAddr = params["model"].asLong();
    if (cfgAddr != 0 && modelAddr != 0)
    {
        Mat &cfgMat = *((Mat *) cfgAddr);
        Mat &modelMat = *((Mat *) modelAddr);
        Mat &classMat = *((Mat *) params["class"].asLong());

        // Load the network config
        std::vector<uchar> vecCfg = (vector<uchar>)(cfgMat);
        std::vector<uchar> vecModel = (vector<uchar>)(modelMat);

        // load class name
        std::vector<char> vecClass = (vector<char>)(classMat);
        splitByLine(vecClass, classes);

        predictNet = dnn::readNetFromDarknet(vecCfg, vecModel);
        predictNet.setPreferableBackend(DNN_BACKEND_OPENCV);
        predictNet.setPreferableTarget(DNN_TARGET_CPU);
    }
    return 0;
}


static long handler_detectObject(cob::ValueMap& params)
{
    long srcAddr = params["src"].asLong();
    long dstAddr = params["dst"].asLong();
    if (srcAddr != 0 && dstAddr != 0)
    {
        Mat &srcMat = *((Mat *) srcAddr);
        Mat &dstMat = *((Mat *) dstAddr);
        Mat frame, blob;

        cvtColor(srcMat, frame, COLOR_BGRA2BGR);

        // Create a 4D blob from a frame.
        blobFromImage(frame, blob, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);

        //Sets the input to the network
        predictNet.setInput(blob);

        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        predictNet.forward(outs, getOutputsNames(predictNet));

        // Remove the bounding boxes with low confidence
        postprocess(frame, outs);

        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = predictNet.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time for a frame : %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
        dstMat = frame;
    }
    return 0;
}

static long handler_processCloak(cob::ValueMap& params)
{
    long srcAddr = params["src"].asLong();
    long dstAddr = params["dst"].asLong();
    long bgAddr = params["bg"].asLong();
    if (srcAddr != 0 && dstAddr != 0) {
        Mat &srcMat = *((Mat *) srcAddr);
        Mat &dstMat = *((Mat *) dstAddr);
        Mat &background = *((Mat *) bgAddr);

        Mat hsv;
        Mat frame = srcMat;
        //flip(frame,frame,1);
        cvtColor(frame, hsv, COLOR_RGB2HSV); //change COLOR_BGR2HSV to COLOR_RGB2HSV

        Mat mask1,mask2;
        inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);
        inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);

        mask1 = mask1 + mask2;

        Mat kernel = Mat::ones(3,3, CV_32F);
        morphologyEx(mask1,mask1,cv::MORPH_OPEN,kernel);
        morphologyEx(mask1,mask1,cv::MORPH_DILATE,kernel);

        bitwise_not(mask1,mask2);

        Mat res1, res2, final_output;
        bitwise_and(frame,frame,res1,mask2);
        bitwise_and(background,background,res2,mask1);
        addWeighted(res1,1,res2,1,0,dstMat);
    }

    return 0;
}

// the 313 ab cluster centers from pts_in_hull.npy (already transposed)
static float hull_pts[] = {
        -90., -90., -90., -90., -90., -80., -80., -80., -80., -80., -80., -80., -80., -70., -70., -70., -70., -70., -70., -70., -70.,
        -70., -70., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -50., -50., -50., -50., -50., -50., -50., -50.,
        -50., -50., -50., -50., -50., -50., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -30.,
        -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -20., -20., -20., -20., -20., -20., -20.,
        -20., -20., -20., -20., -20., -20., -20., -20., -20., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10.,
        -10., -10., -10., -10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 10., 10., 10., 10., 10., 10.,
        10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.,
        20., 20., 20., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 40., 40., 40., 40.,
        40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50.,
        50., 50., 50., 50., 50., 50., 50., 50., 50., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60.,
        60., 60., 60., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 80., 80., 80.,
        80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 90., 90., 90., 90., 90., 90., 90., 90., 90., 90.,
        90., 90., 90., 90., 90., 90., 90., 90., 90., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 50., 60., 70., 80., 90.,
        20., 30., 40., 50., 60., 70., 80., 90., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -20., -10., 0., 10., 20., 30., 40., 50.,
        60., 70., 80., 90., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -40., -30., -20., -10., 0., 10., 20.,
        30., 40., 50., 60., 70., 80., 90., 100., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -50.,
        -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -60., -50., -40., -30., -20., -10., 0., 10., 20.,
        30., 40., 50., 60., 70., 80., 90., 100., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90.,
        100., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -80., -70., -60., -50.,
        -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -90., -80., -70., -60., -50., -40., -30., -20., -10.,
        0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30.,
        40., 50., 60., 70., 80., 90., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70.,
        80., -110., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., -110., -100.,
        -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., -110., -100., -90., -80., -70.,
        -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., -110., -100., -90., -80., -70., -60., -50., -40., -30.,
        -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0.
};

static Net colorization_net;

static long handler_colorization_configNet(cob::ValueMap& params)
{
    auto cfgAddr = params["cfg"].asLong();
    auto modelAddr = params["model"].asLong();
    if (cfgAddr != 0 && modelAddr != 0)
    {
        Mat &cfgMat = *((Mat *) cfgAddr);
        Mat &modelMat = *((Mat *) modelAddr);

        // Load the network config
        std::vector<uchar> vecCfg = (vector<uchar>)(cfgMat);
        std::vector<uchar> vecModel = (vector<uchar>)(modelMat);

        colorization_net = dnn::readNetFromCaffe(vecCfg, vecModel);

        // setup additional layers:
        int sz[] = {2, 313, 1, 1};
        const Mat pts_in_hull(4, sz, CV_32F, hull_pts);
        Ptr<dnn::Layer> class8_ab = colorization_net.getLayer("class8_ab");
        class8_ab->blobs.push_back(pts_in_hull);
        Ptr<dnn::Layer> conv8_313_rh = colorization_net.getLayer("conv8_313_rh");
        conv8_313_rh->blobs.push_back(Mat(1, 313, CV_32F, Scalar(2.606)));
    }
    return 0;
}

static long handler_colorize(cob::ValueMap& params)
{
    const int W_in = 224;
    const int H_in = 224;

    long srcAddr = params["src"].asLong();
    long dstAddr = params["dst"].asLong();
    if (srcAddr != 0 && dstAddr != 0)
    {
        Mat &srcMat = *((Mat *) srcAddr);
        Mat &dstMat = *((Mat *) dstAddr);
        Mat frame = srcMat;

        // extract L channel and subtract mean
        Mat lab, L, input;
        frame.convertTo(frame, CV_32F, 1.0/255);
        cvtColor(frame, lab, COLOR_RGB2Lab);
        extractChannel(lab, L, 0);
        resize(L, input, Size(W_in, H_in));
        input -= 50;

        // run the L channel through the network
        Mat inputBlob = blobFromImage(input);
        colorization_net.setInput(inputBlob);
        Mat result = colorization_net.forward();

        // retrieve the calculated a,b channels from the network output
        Size siz(result.size[2], result.size[3]);
        Mat a = Mat(siz, CV_32F, result.ptr(0,0));
        Mat b = Mat(siz, CV_32F, result.ptr(0,1));
        resize(a, a, frame.size());
        resize(b, b, frame.size());

        // merge, and convert back to BGR
        Mat &color = dstMat;
        Mat chn[] = {L, a, b};
        merge(chn, 3, lab);
        cvtColor(lab, color, COLOR_Lab2RGB);

        color = color*255;
        color.convertTo(color, CV_8U);
    }
    return 0;
}

//// age gender ////
static Net ag_ageNet;
static Net ag_genderNet;
static Net ag_faceNet;

static long handler_ageGender_configNet(cob::ValueMap& params)
{
    auto& cfgArray = params["cfg"].asValueVector();
    auto& modelArray = params["model"].asValueVector();
    if (cfgArray.size() == 3 && modelArray.size() == 3)
    {
        Mat &ageCfg = *((Mat *) cfgArray[0].asLong());
        Mat &genderCfg = *((Mat *) cfgArray[1].asLong());
        Mat &faceCfg = *((Mat *) cfgArray[2].asLong());
        Mat &ageModel = *((Mat *) modelArray[0].asLong());
        Mat &genderModel = *((Mat *) modelArray[1].asLong());
        Mat &faceModel = *((Mat *) modelArray[2].asLong());

        // Load Network
        ag_ageNet = dnn::readNetFromCaffe((vector<uchar>)(ageCfg), (vector<uchar>)(ageModel));
        ag_genderNet = dnn::readNetFromCaffe((vector<uchar>)(genderCfg), (vector<uchar>)(genderModel));
        ag_faceNet = dnn::readNetFromTensorflow((vector<uchar>)(faceModel), (vector<uchar>)(faceCfg));
    }
    return 0;
}

tuple<Mat, vector<vector<int>>> getFaceBox(Net net, Mat &frame, double conf_threshold)
{
    Mat frameOpenCVDNN = frame.clone();
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
    double inScaleFactor = 1.0;
    Size size = Size(300, 300);
    // std::vector<int> meanVal = {104, 117, 123};
    Scalar meanVal = Scalar(104, 117, 123);

    cv::Mat inputBlob;
    //inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, size, meanVal, true, false);
    inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(frameWidth, frameHeight), meanVal, true, false);

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    vector<vector<int>> bboxes;

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > conf_threshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
            vector<int> box = {x1, y1, x2, y2};
            bboxes.push_back(box);
            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
        }
    }

    return make_tuple(frameOpenCVDNN, bboxes);
}

static long handler_detectAgeGender(cob::ValueMap& params)
{
    const int padding = 20;
    Scalar MODEL_MEAN_VALUES = Scalar(78.4263377603, 87.7689143744, 114.895847746);
    vector<string> ageList = {"(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
                              "(38-43)", "(48-53)", "(60-100)"};

    vector<string> genderList = {"Male", "Female"};

    long srcAddr = params["src"].asLong();
    long dstAddr = params["dst"].asLong();
    if (srcAddr != 0 && dstAddr != 0)
    {
        Mat &srcMat = *((Mat *) srcAddr);
        Mat &dstMat = *((Mat *) dstAddr);
        Mat frame = srcMat;

        vector<vector<int>> bboxes;
        Mat frameFace;
        tie(frameFace, bboxes) = getFaceBox(ag_faceNet, frame, 0.7);

        if(bboxes.size() > 0)
        {
            for (auto it = begin(bboxes); it != end(bboxes); ++it)
            {
                Rect rec(it->at(0) - padding, it->at(1) - padding,
                         it->at(2) - it->at(0) + 2 * padding,
                         it->at(3) - it->at(1) + 2 * padding);
                Mat face = frame(rec); // take the ROI of box on the frame

                Mat blob;
                blob = blobFromImage(face, 1, Size(227, 227), MODEL_MEAN_VALUES, false);
                ag_genderNet.setInput(blob);
                // string gender_preds;
                vector<float> genderPreds = ag_genderNet.forward();
                // printing gender here
                // find max element index
                // distance function does the argmax() work in C++
                int max_index_gender = std::distance(genderPreds.begin(),
                                                     max_element(genderPreds.begin(),
                                                                 genderPreds.end()));
                string gender = genderList[max_index_gender];

                ag_ageNet.setInput(blob);
                vector<float> agePreds = ag_ageNet.forward();
                /* // uncomment below code if you want to iterate through the age_preds
                 * vector
                cout << "PRINTING AGE_PREDS" << endl;
                for(auto it = age_preds.begin(); it != age_preds.end(); ++it) {
                  cout << *it << endl;
                }
                */

                // finding maximum indicd in the age_preds vector
                int max_indice_age = std::distance(agePreds.begin(),
                                                   max_element(agePreds.begin(), agePreds.end()));
                string age = ageList[max_indice_age];
                string label = gender + ", " + age; // label
                cv::putText(frameFace, label, Point(it->at(0), it->at(1) - 15),
                            cv::FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 255), 2, cv::LINE_AA);
            }
        }
    }
    return 0;
}


static std::unordered_map<std::string, std::function<long(cob::ValueMap&)>> mNativeCallHandlerMap {
    {"cvt2Gray", handler_convertToGray},
    {"drawRect", handler_drawRect},
    {"detectBlob", handler_detectBlob},
    {"configNet", handler_configNet},
    {"detectObject", handler_detectObject},
    {"processCloak", handler_processCloak},
    {"colorization_configNet", handler_colorization_configNet},
    {"colorize", handler_colorize},
    {"ageGender_configNet", handler_ageGender_configNet},
    {"detectAgeGender", handler_detectAgeGender},
};

JNIEXPORT long JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeCallCxx(JNIEnv*  env, jclass thiz, jstring jstrFunc, jstring jstrParam)
{
    std::string funcName = env->GetStringUTFChars(jstrFunc, NULL);
    std::string jsonParams = env->GetStringUTFChars(jstrParam, NULL);
    LOGD("nativeCallCxx func: %s, param: %s", funcName.c_str(), jsonParams.c_str());

    auto iter = mNativeCallHandlerMap.find(funcName);
    if (iter != mNativeCallHandlerMap.end())
    {
        cob::ValueMap vm;
        vm.fromJson(jsonParams);
        auto& handler = iter->second;
        return handler(vm);
    }
    return 0;
}