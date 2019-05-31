package org.opencv.samples.facedetect;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;
import android.widget.Toast;

import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONArray;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class AgeGenderActivity extends AppCompatActivity implements CvCameraViewListener2 {

    private static final String    TAG                 = "OCVSample::Activity";
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(255, 0, 0, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;

    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;

    private Mat                    mRgba;
    private Mat                    mGray;
    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;

    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;

    enum Status {PREVIEW, SNAPSHOT, COLORIZED}

    //
    private Bitmap mSrcBitmap = null;
    private Mat mSrcbMat;
    private Mat mDstbMat;
    private Mat mAgeCfgMat;
    private Mat mAgeModelMat;
    private Mat mGenderCfgMat;
    private Mat mGenderModelMat;
    private Mat mFaceCfgMat;
    private Mat mFaceModelMat;
    private boolean mNetConfigured = false;
    private Status mStatus = Status.PREVIEW;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public AgeGenderActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();

        if (mSrcBitmap == null) {
            mSrcbMat = new Mat();
            mDstbMat = new Mat();
            mAgeCfgMat = new Mat();
            mAgeModelMat = new Mat();
            mGenderCfgMat = new Mat();
            mGenderModelMat = new Mat();
            mFaceCfgMat = new Mat();
            mFaceModelMat = new Mat();

            mSrcBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.mandog);
            Utils.bitmapToMat(mSrcBitmap, mSrcbMat);
        }

        /*
        if (!mNetConfigured) {
            Toast showToast = Toast.makeText(AgeGenderActivity.this, "Camera is starting...", Toast.LENGTH_LONG);
            showToast.setGravity(Gravity.BOTTOM, 0, 0);
            showToast.show();

            new Thread(new Runnable() {
                @Override
                public void run() {
                    configDetectNet();
                    mNetConfigured = true;
                }
            }, "configNet").start();
        }*/
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();

        //mSrcbMat.release();
        //mDstbMat.release();
        mAgeCfgMat.release();
        mAgeModelMat.release();
        mGenderCfgMat.release();
        mGenderModelMat.release();
        mFaceCfgMat.release();
        mFaceModelMat.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        if (mStatus == Status.PREVIEW) {
            inputFrame.rgba().copyTo(mSrcbMat);
            return mSrcbMat;
        } else if (mStatus == Status.SNAPSHOT) {
            return mSrcbMat;
        } else {
            return mDstbMat;
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");

        mItemFace40 = menu.add("Detect Age & Gender");
        mItemFace30 = menu.add("Preview");
        //mItemFace50 = menu.add("Take photo");
        //mItemFace20 = menu.add("bird");
        mItemType   = menu.add("config net");
        return true;
    }

    public Mat inputStream2Mat(InputStream in) throws IOException {
        byte[] b = new byte[64*1024*1024];
        int count = 0;
        int n = 0;
        while ((n = in.read(b, count, b.length - count)) != -1) {
            count += n;
        }
        Mat mat = new Mat(1, count, CvType.CV_8UC1);
        mat.put(0, 0, b);
        return mat;
    }

    public void configDetectNet() {
        JSONObject jsonObject = new JSONObject();
        try {
            if (mAgeCfgMat.empty()) {
                InputStream in = getResources().openRawResource(R.raw.age_deploy_prototxt);
                mAgeCfgMat = inputStream2Mat(in);
            }
            if (mGenderCfgMat.empty()) {
                InputStream in = getResources().openRawResource(R.raw.gender_deploy_prototxt);
                mGenderCfgMat = inputStream2Mat(in);
            }
            if (mFaceCfgMat.empty()) {
                InputStream in = getResources().openRawResource(R.raw.opencv_face_detector_pbtxt);
                mFaceCfgMat = inputStream2Mat(in);
            }
            JSONArray cfgArray = new JSONArray();
            cfgArray.put(mAgeCfgMat.getNativeObjAddr());
            cfgArray.put(mGenderCfgMat.getNativeObjAddr());
            cfgArray.put(mFaceCfgMat.getNativeObjAddr());
            jsonObject.put("cfg", cfgArray);

            if (mAgeModelMat.empty()) {
                InputStream in = getResources().openRawResource(R.raw.age_net_caffemodel);
                mAgeModelMat =  inputStream2Mat(in);
            }
            if (mGenderModelMat.empty()) {
                InputStream in = getResources().openRawResource(R.raw.gender_net_caffemodel);
                mGenderModelMat =  inputStream2Mat(in);
            }
            if (mFaceModelMat.empty()) {
                InputStream in = getResources().openRawResource(R.raw.opencv_face_detector_uint8_pb);
                mFaceModelMat =  inputStream2Mat(in);
            }
            JSONArray modelArray = new JSONArray();
            modelArray.put(mAgeModelMat.getNativeObjAddr());
            modelArray.put(mGenderModelMat.getNativeObjAddr());
            modelArray.put(mFaceModelMat.getNativeObjAddr());
            jsonObject.put("model", modelArray);
        } catch (JSONException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        mNativeDetector.callCxx("ageGender_configNet", jsonObject.toString());
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        if (item == mItemFace50) {
            mStatus = Status.SNAPSHOT;
        }

        if (item != mItemType && !mNetConfigured)
            return false;

        if (item == mItemFace40) {
            mStatus = Status.SNAPSHOT;

            Toast showToast = Toast.makeText(AgeGenderActivity.this, "Detecting...", Toast.LENGTH_LONG);
            showToast.setGravity(Gravity.BOTTOM, 0, 0);
            showToast.show();

            new Thread(new Runnable() {
                @Override
                public void run() {
                    JSONObject jsonObject = new JSONObject();
                    try {
                        jsonObject.put("src", mSrcbMat.getNativeObjAddr());
                        jsonObject.put("dst", mDstbMat.getNativeObjAddr());
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }
                    mNativeDetector.callCxx("detectAgeGender", jsonObject.toString());
                    mStatus = Status.COLORIZED;
                }
            }, "colorize").start();
        }
        else if (item == mItemFace30) {
            mStatus = Status.PREVIEW;
        }
        else if (item == mItemType) {
            configDetectNet();
            mNetConfigured = true;
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
    }
}
