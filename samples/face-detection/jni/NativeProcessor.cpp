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

///////////////// native call hanlder ////////////////////

static long hanlder_convertToGray(cob::ValueMap& params)
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

static long hanlder_drawRect(cob::ValueMap& params)
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

static long hanlder_detectBlob(cob::ValueMap& params)
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


static long hanlder_configNet(cob::ValueMap& params)
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

        predictNet = readNetFromDarknet(vecCfg, vecModel);
        predictNet.setPreferableBackend(DNN_BACKEND_OPENCV);
        predictNet.setPreferableTarget(DNN_TARGET_CPU);
    }
    return 0;
}


static long hanlder_detectObject(cob::ValueMap& params)
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

static long hanlder_processCloak(cob::ValueMap& params)
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

static std::unordered_map<std::string, std::function<long(cob::ValueMap&)>> mNativeCallHandlerMap {
    {"cvt2Gray", hanlder_convertToGray},
    {"drawRect", hanlder_drawRect},
    {"detectBlob", hanlder_detectBlob},
    {"configNet", hanlder_configNet},
    {"detectObject", hanlder_detectObject},
    {"processCloak", hanlder_processCloak},
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