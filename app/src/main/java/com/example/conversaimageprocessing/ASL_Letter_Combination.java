package com.example.conversaimageprocessing;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;

public class ASL_Letter_Combination extends Activity implements CameraBridgeViewBase.CvCameraViewListener2{
    // Tag used for Logging Error Messages
    private static final String TAG="HomeScreen";
    // Matrices for processing the frames
    private Mat RGBaMatrix;
    private Mat GrayMatrix;
    // Camera View
    private CameraBridgeViewBase OpenCVCameraView;
    // Translation class for recognising Sign Language from a hand
    private SignLanguageTranslationClass SignLanguageTranslationClass;
    // Buttons and TextView for the UI
    private Button add_button, clear_button, backspace_button, enter_button;
    private TextView change_Text;
    // Callback for loading the OpenCV library
    private BaseLoaderCallback baseLoaderCallback =new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface.SUCCESS:
                {
                    // If OpenCV has successfully loaded
                    Log.i(TAG,"OpenCV Is loaded");
                    OpenCVCameraView.enableView();
                }
                default:
                {
                    super.onManagerConnected(status);

                }
                break;
            }
        }
    };

    public ASL_Letter_Combination(){
        Log.i(TAG,"Instantiated new "+this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Hide the title bar and keep the screen on at all times
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        // Variable to check if camera permission has been granted
        int CAMERA_PERMISSION_REQUEST=0;
        // If this is not true, request permission
        if (ContextCompat.checkSelfPermission(ASL_Letter_Combination.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(ASL_Letter_Combination.this, new String[] {Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
        }
        //Set the layout for the screen as the XML file
        setContentView(R.layout.activity_asl_letter_combination);

        // Initialize the camera view and listener
        OpenCVCameraView=(CameraBridgeViewBase) findViewById(R.id.java_camera_view);
        OpenCVCameraView.setVisibility(SurfaceView.VISIBLE);
        OpenCVCameraView.setCvCameraViewListener(this);
        // Find the buttons and text views using findViewById
        add_button = findViewById(R.id.add_button);
        clear_button = findViewById(R.id.clear_button);
        backspace_button = findViewById(R.id.backspace_button);
        enter_button = findViewById(R.id.enter_button);
        change_Text = findViewById(R.id.change_text);
        try {
            // Initialize SignLanguageTranslationClass with required parameters
            SignLanguageTranslationClass = new SignLanguageTranslationClass(add_button, backspace_button, clear_button, change_Text, getAssets(), "hand_model.tflite", "letter_list.txt", 300, "Sign_language_model.tflite", 96);
            // Set click listener for enter button to send translated text back to Conversa
            enter_button.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    // Get the translated text from the text view at the appropriate time
                    String ASL_Translated_Text = change_Text.getText().toString();
                    // Create an intent to return data to the previous activity (Chat.java)
                    Intent intent = new Intent();
                    intent.putExtra("testMessage", ASL_Translated_Text); // Pass the test message as an extra with key "testMessage"
                    setResult(Activity.RESULT_OK, intent);
                    finish(); // Finish current activity
                }
            });
            Log.d("HomeScreen", "Model is successfully loaded"); // Log success message
        } catch (IOException e) {
            Log.d("HomeScreen", "Getting some error"); // Log error message
            e.printStackTrace();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Check if OpenCV library has been initialized successfully
        if (OpenCVLoader.initDebug()){
            //if successful, load the message and connect to the manager
            Log.d(TAG,"Opencv initialization is done");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else{
            //if unsuccessful, log the message and try asynchronous initialization
            Log.d(TAG,"Opencv is not loaded. try again");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION,this,baseLoaderCallback);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        // Check if the camera view is initialized
        if (OpenCVCameraView !=null){
            // if so, disable the camera view
            OpenCVCameraView.disableView();
        }
    }
    // onDestroy method called when the activity is destroyed
    public void onDestroy(){
        super.onDestroy();
        // Check if the camera view is initialized
        if(OpenCVCameraView !=null){
            // If initialized, disable the camera view
            OpenCVCameraView.disableView();
        }
    }
    // Callback method invoked when the camera view is started
    public void onCameraViewStarted(int width ,int height){
        // Create a new RGBA matrix with specific width and height
        RGBaMatrix = new Mat(height,width, CvType.CV_8UC4);
        // Make a greyscale matrix with the same parameters
        GrayMatrix = new Mat(height,width,CvType.CV_8UC1);
    }
    // Callback method is called when the camera view has been stopped
    public void onCameraViewStopped(){
        // Release the resources for the RGBA matrix
        RGBaMatrix.release();
    }
    // Callback Method called when a new camera view is now available
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
        // Retrieve both the RGBA and Grey matrices from the camera frame
        RGBaMatrix = inputFrame.rgba();
        GrayMatrix = inputFrame.gray();
        // Create a new output matrix
        Mat ASL_Translated_Matrix = new Mat();
        // Perform image recognition using the SignLanguageTranslationClass
        ASL_Translated_Matrix = SignLanguageTranslationClass.recognizeImage(RGBaMatrix);
        // Return the matrix
        return ASL_Translated_Matrix;
    }
}