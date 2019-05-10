package org.opencv.samples.facedetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

import org.json.JSONObject;
import org.json.JSONException;

public class FdActivity extends AppCompatActivity implements CvCameraViewListener2 {

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

    //
    private Bitmap mSrcBitmap = null;
    private Mat mSrcbMat;
    private Mat mDstbMat;
    private Mat mCanvasMat;
    private Mat mCfgbMat;
    private Mat mModelMat;
    private Mat mClassMat;
    private boolean mNetConfigured = false;

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

    public FdActivity() {
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
            mCfgbMat = new Mat();
            mModelMat = new Mat();
            mClassMat = new Mat();

            mSrcBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.mandog);
            Utils.bitmapToMat(mSrcBitmap, mSrcbMat);
        }
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();

        //mSrcbMat.release();
        //mDstbMat.release();
        mCfgbMat.release();
        mModelMat.release();
        mClassMat.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        if (mDstbMat.empty())
            return mSrcbMat;
        else
            return mDstbMat;

        /*
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }

        MatOfRect faces = new MatOfRect();

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 0, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null)
                mNativeDetector.detect(mGray, faces);
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
        }

        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++)
            Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);

        return mRgba;*/
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("people horse");
        mItemFace40 = menu.add("street");
        mItemFace30 = menu.add("man dog");
        mItemFace20 = menu.add("bird");
        mItemType   = menu.add("config net");
        return true;
    }

    public Mat inputStream2Mat(InputStream in) throws IOException {
        byte[] b = new byte[256*1024*1024];
        int count = 0;
        int n = 0;
        while ((n = in.read(b, count, b.length - count)) != -1) {
            count += n;
        }
        Mat mat = new Mat(1, count, CvType.CV_8UC1);
        mat.put(0, 0, b);
        return mat;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        if (item != mItemType && !mNetConfigured)
            return false;

        if (item == mItemFace50) {
            Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.people_horse);
            Utils.bitmapToMat(bitmap, mSrcbMat);

            JSONObject jsonObject = new JSONObject();
            try {
                jsonObject.put("src", mSrcbMat.getNativeObjAddr());
                jsonObject.put("dst", mDstbMat.getNativeObjAddr());
            } catch (JSONException e) {
                e.printStackTrace();
            }
            mNativeDetector.callCxx("detectObject", jsonObject.toString());
        }
        else if (item == mItemFace40) {
            Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.street);
            Utils.bitmapToMat(bitmap, mSrcbMat);

            JSONObject jsonObject = new JSONObject();
            try {
                jsonObject.put("src", mSrcbMat.getNativeObjAddr());
                jsonObject.put("dst", mDstbMat.getNativeObjAddr());
            } catch (JSONException e) {
                e.printStackTrace();
            }
            mNativeDetector.callCxx("detectObject", jsonObject.toString());
        }
        else if (item == mItemFace30) {
            Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.mandog);
            Utils.bitmapToMat(bitmap, mSrcbMat);

            JSONObject jsonObject = new JSONObject();
            try {
                jsonObject.put("src", mSrcbMat.getNativeObjAddr());
                jsonObject.put("dst", mDstbMat.getNativeObjAddr());
            } catch (JSONException e) {
                e.printStackTrace();
            }
            mNativeDetector.callCxx("detectObject", jsonObject.toString());
        }
        else if (item == mItemFace20) {
            Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.bird);
            Utils.bitmapToMat(bitmap, mSrcbMat);

            JSONObject jsonObject = new JSONObject();
            try {
                jsonObject.put("src", mSrcbMat.getNativeObjAddr());
                jsonObject.put("dst", mDstbMat.getNativeObjAddr());
            } catch (JSONException e) {
                e.printStackTrace();
            }
            mNativeDetector.callCxx("detectObject", jsonObject.toString());
        }
        else if (item == mItemType) {
            JSONObject jsonObject = new JSONObject();
            try {
                if (mCfgbMat.empty()) {
                    InputStream cfgIn = getResources().openRawResource(R.raw.yolov3_cfg);
                    mCfgbMat = inputStream2Mat(cfgIn);
                }
                jsonObject.put("cfg", mCfgbMat.getNativeObjAddr());

                if (mModelMat.empty()) {
                    InputStream weightsIn = getResources().openRawResource(R.raw.yolov3_weights);
                    mModelMat =  inputStream2Mat(weightsIn);
                }
                jsonObject.put("model", mModelMat.getNativeObjAddr());

                if (mClassMat.empty()) {
                    InputStream classIn = getResources().openRawResource(R.raw.coco_names);
                    mClassMat =  inputStream2Mat(classIn);
                }
                jsonObject.put("class", mClassMat.getNativeObjAddr());
            } catch (JSONException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
            mNativeDetector.callCxx("configNet", jsonObject.toString());
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
