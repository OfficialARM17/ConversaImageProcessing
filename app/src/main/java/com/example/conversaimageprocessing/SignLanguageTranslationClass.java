package com.example.conversaimageprocessing;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class SignLanguageTranslationClass {
    // this is used to load model and predict
    private Interpreter hand_model_interpreter;
    //create another interpreter needed to understand sign language from the hand
    private Interpreter sign_language_model_interpreter;
    // store all labels in an String list which corresponds for each letter
    private List<String> letterList;
    private int image_height=0;
    private int image_width=0;
    //The input size for the model
    private int INPUT_SIZE;
    private int PIXEL_SIZE=3; // for RGB
    private int IMAGE_MEAN=0;
    private  float IMAGE_STD=255.0f;
    // GPU delegate for processing imagery in real-time
    private GpuDelegate gpu_Delegate;
    private int Classification_Input_Size=0;
    // Strings used to manipulate the Text View
    private String final_text="";
    private String current_text="";
    // Constructor of the SignLanguageTranslationClass
    SignLanguageTranslationClass(Button add_button, Button backspace_button, Button clear_button, TextView change_text, AssetManager assetManager, String modelPath, String labelPath, int inputSize, String classification_model, int classification_input_size) throws IOException{
        // Set the input size variable for the main model
        INPUT_SIZE=inputSize;
        // Set the input size variable for the sign language model
        Classification_Input_Size=classification_input_size;
        // Settings the options for the main model interpreter
        Interpreter.Options hand_model_interpreter_options = new Interpreter.Options();
        gpu_Delegate=new GpuDelegate();
        hand_model_interpreter_options.addDelegate(gpu_Delegate);
        hand_model_interpreter_options.setNumThreads(8); // Can change the number of threads depending on the user's phone
        // loading the main model for detecting a hand from the background
        hand_model_interpreter=new Interpreter(loadHandModelFile(assetManager,modelPath),hand_model_interpreter_options);
        // loading the label map that the hand model requires
        letterList=loadLetterList(assetManager,labelPath);
        // Confguring the settings for the sign language model interpreter
        Interpreter.Options sign_language_model_interpreter_options =new Interpreter.Options();
        // set the number of threads to 8 to improve accuracy
        sign_language_model_interpreter_options.setNumThreads(8);
        // Loading the sign language model
        sign_language_model_interpreter=new Interpreter(loadHandModelFile(assetManager, classification_model),sign_language_model_interpreter_options);
        // add Button will add the value of current_text onto the text view
        add_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                final_text=final_text+current_text;
                // the text view will then be updated with the new value added
                change_text.setText(final_text);
            }
        });
        // backspace_button will remove the last added value onto final_text
        backspace_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Check if there's any text to remove to begin with
                if (!final_text.isEmpty()) {
                    // Remove the last letter from final_text
                    final_text = final_text.substring(0, final_text.length() - 1);
                    // update the textview with the button press with the new value
                    change_text.setText(final_text);
                }
            }
        });
        // clear button will remove all text from the textview
        clear_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // set the value of final_text to nothing
                final_text="";
                // change the text view to final_text
                change_text.setText(final_text);
            }
        });
    }
    //Loads a list of letters from the provided file
    private List<String> loadLetterList(AssetManager assetManager, String letterPath) throws IOException {
        // Create a list to store the loaded letters
        List<String> letterList=new ArrayList<>();
        // Create a reader to read the letter file
        BufferedReader reader=new BufferedReader(new InputStreamReader(assetManager.open(letterPath)));
        String line;
        // Loop through each line in the letter file and add it to the letter list
        while ((line=reader.readLine())!=null){
            letterList.add(line);
        }
        reader.close(); // Close the reader
        return letterList; // Return the list of loaded letters
    }
    // Load the hand model from the provided asset path
    private ByteBuffer loadHandModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // Retrieve the file descriptor of the model file
        AssetFileDescriptor fileDescriptor=assetManager.openFd(modelPath);
        // Create an input stream to read from the file descriptor
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        // Retrieve the file channel from the input stream
        FileChannel fileChannel=inputStream.getChannel();
        // Retrieve the start offset and declared length of the model file
        long startOffset =fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();
        // Map the file channel to a byte buffer
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }
    // Recognize objects in the input image using the model
    public Mat recognizeImage(Mat mat_image){
        // Rotate the original image 90 degrees to make it portrait
        // This is needed to prevent crashing issues
        Mat rotated_mat_image=new Mat();
        Mat a=mat_image.t(); // Transpose the image
        Core.flip(a,rotated_mat_image,1); // Flip the transposed image horizontally
        // Release the matrix
        a.release();

        // Convert the rotated image matrix to bitmap
        Bitmap bitmap=Bitmap.createBitmap(rotated_mat_image.cols(),rotated_mat_image.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotated_mat_image,bitmap);

        // define the image's height and width
        image_height=bitmap.getHeight();
        image_width=bitmap.getWidth();

        // Scale the bitmap to the input size of the model
        Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);

        // Convert the bitmap to a byte buffer as the model input
        ByteBuffer byteBuffer=handModelBitmapToByteBuffer(scaledBitmap);

        // Define output variables for the model predictions
        Object[] input=new Object[1];
        input[0]=byteBuffer;

        Map<Integer,Object> output_map=new TreeMap<>();
        float[][][]boxes =new float[1][10][4]; // Stores bounding box coordinates
        float[][] scores=new float[1][10]; // Stores object scores
        float[][] classes=new float[1][10]; // Stores object classes

        // Add output arrays to the output map
        output_map.put(0,boxes);
        output_map.put(1,classes);
        output_map.put(2,scores);

        // Perform predicition using the hand model interpreter
        hand_model_interpreter.runForMultipleInputsOutputs(input,output_map);

        // Extract predicted values from the output map
        Object value=output_map.get(0);
        Object Object_class=output_map.get(1);
        Object score=output_map.get(2);

        // loop through each detected object
        for (int i=0;i<10;i++){
            // Extract class and score values for this object
            float class_value=(float) Array.get(Array.get(Object_class,0),i);
            float score_value=(float) Array.get(Array.get(score,0),i);
            // Define threshold for score
            // Adjust threshold as needed for the model
            if(score_value>0.5){
                Object box1=Array.get(Array.get(value,0),i);
                // Multiply bounding box coordinates by original height and width of the frame
                float y1=(float) Array.get(box1,0)*image_height;
                float x1=(float) Array.get(box1,1)*image_width;
                float y2=(float) Array.get(box1,2)*image_height;
                float x2=(float) Array.get(box1,3)*image_width;
                // Set the boundary limits for the coordinates
                if(y1<0){ y1=0;}
                if(x1<0){ x1=0;}
                if(y2>image_height){ y2=image_height;}
                if(x2>image_width){ x2=image_width; }
                // Calculate the height and width of the box
                float w1=x2-x1;
                float h1=y2-y1;
                // Crop the hand image
                Rect cropped_roi = new Rect((int)x1,(int)y1,(int)w1,(int)h1);
                Mat cropped=new Mat(rotated_mat_image,cropped_roi).clone();
                // Convert the cropped matrix to a bitmap
                Bitmap bitmap1=Bitmap.createBitmap(cropped.cols(),cropped.rows(),Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(cropped,bitmap1);
                // Resize bitmap1 to 96x96
                Bitmap scaledBitmap1=Bitmap.createScaledBitmap(bitmap,Classification_Input_Size,Classification_Input_Size,false);
                ByteBuffer byteBuffer1=signLanguageModelBitmapToByteBuffer1(scaledBitmap1);
                // Create an array for signlanguageinterpreter output
                float[][] output_class_value = new float[1][1];
                // Perform prediction for byteBuffer1 using the sign language model interpreter
                sign_language_model_interpreter.run(byteBuffer1,output_class_value);
                // Use the calculated value from the calculate_Alphabet_Value class and assign the value to current_text fot the text view
                String sign_val=calculate_Alphabet_Value(output_class_value[0][0]);
                current_text=sign_val;
                // Add class name in the image
                Imgproc.putText(rotated_mat_image,""+sign_val,new Point(x1+10,y1+40),2,1.5,new Scalar(255, 255, 255, 255),2);
                // Draw a rectangle around the detected object
                Imgproc.rectangle(rotated_mat_image,new Point(x1,y1),new Point(x2,y2),new Scalar(0, 255, 0, 255),5);
            }
        }
        // Transpose and flip the rotated_mat_image to original orientation
        Mat b=rotated_mat_image.t();
        Core.flip(b,mat_image,0);
        b.release();
        return mat_image;   // Return the processed iamge matrix
    }
    // Method to convert a float value to alphabet values depending on predefined ranges
    private String calculate_Alphabet_Value(float ASL_Value) {
        String val="";
        // List of checks of the vlaue of ASL_Value to se
        if(ASL_Value>=-0.5 & ASL_Value<0.5){
            val="A";
        }
        else if(ASL_Value>=0.5 & ASL_Value<1.5){
            val="B";
        }
        else if(ASL_Value>=1.5 & ASL_Value<2.5){
            val="C";
        }
        else if(ASL_Value>=2.5 & ASL_Value<3.5){
            val="D";
        }
        else if(ASL_Value>=3.5 & ASL_Value<4.5){
            val="E";
        }
        else if(ASL_Value>=4.5 & ASL_Value<5.5){
            val="F";
        }
        else if(ASL_Value>=5.5 & ASL_Value<6.5){
            val="G";
        }
        else if(ASL_Value>=6.5 & ASL_Value<7.5){
            val="H";
        }
        else if(ASL_Value>=7.5 & ASL_Value<8.5){
            val="I";
        }
        else if(ASL_Value>8.5 & ASL_Value<9.5){
            val="J";
        }
        else if(ASL_Value>=9.5 & ASL_Value<10.5){
            val="K";
        }
        else if(ASL_Value>=10.5 & ASL_Value<11.5){
            val="L";
        }
        else if(ASL_Value>=11.5 & ASL_Value<12.5){
            val="M";
        }
        else if(ASL_Value>=12.5 & ASL_Value<13.5){
            val="N";
        }
        else if(ASL_Value>=13.5 & ASL_Value<14.5){
            val="O";
        }
        else if(ASL_Value>=14.5 & ASL_Value<15.5){
            val="P";
        }
        else if(ASL_Value>=15.5 & ASL_Value<16.5){
            val="Q";
        }
        else if(ASL_Value>=16.5 & ASL_Value<17.5){
            val="R";
        }
        else if(ASL_Value>=17.5 & ASL_Value<18.5){
            val="S";
        }
        else if(ASL_Value>=18.5 & ASL_Value<19.5){
            val="T";
        }
        else if(ASL_Value>=19.5 & ASL_Value<20.5){
            val="U";
        }
        else if(ASL_Value>=20.5 & ASL_Value<21.5){
            val="V";
        }
        else if(ASL_Value>=21.5 & ASL_Value<22.5){
            val="W";
        }
        else if(ASL_Value>=22.5 & ASL_Value<23.5){
            val="X";
        }
        else{
            val="Y";
        }
        return val;
    }

    private ByteBuffer handModelBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        // Determine whether the model input should be quantized
        // 0 for no quantization, 1 for quantization
        // Change the value of quantization_value on the need
        // Since the image is being scaled, quantization is needed
        int quantization_value=1;
        int size_of_images=INPUT_SIZE;
        if(quantization_value==0){
            // Allocate byte buffer based on the size of the image and RGB channels
            byteBuffer=ByteBuffer.allocateDirect(1*size_of_images*size_of_images*3);
        }
        else {
            // Allocate byte buffer considering float values and RGB channels
            byteBuffer=ByteBuffer.allocateDirect(4*1*size_of_images*size_of_images*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());

        // Extract pixel values from the bitmap
        int[] intValues=new int[size_of_images*size_of_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;

        // Iterate over each pixel in the image
        for (int i=0;i<size_of_images;++i){
            for (int j=0;j<size_of_images;++j){
                final  int val=intValues[pixel++];
                if(quantization_value==0){
                    // Convert RGB values to byte for no quantization
                    byteBuffer.put((byte) ((val>>16)&0xFF));
                    byteBuffer.put((byte) ((val>>8)&0xFF));
                    byteBuffer.put((byte) (val&0xFF));
                }
                else {
                    // Convert RGB values to float for quantization
                    byteBuffer.putFloat((((val >> 16) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val >> 8) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val) & 0xFF))/255.0f);
                }
            }
        }
        return byteBuffer;
    }
    // Same functionality as handModelBitmapToByteBuffer but the size_of_images
    // is using the required value for the sign language interpreter
    private ByteBuffer signLanguageModelBitmapToByteBuffer1(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        int quantization_value=1;
        //change input size for sign_language_model_interpreter
        int size_of_images=Classification_Input_Size;
        if(quantization_value==0){
            byteBuffer=ByteBuffer.allocateDirect(1*size_of_images*size_of_images*3);
        }
        else {
            byteBuffer=ByteBuffer.allocateDirect(4*1*size_of_images*size_of_images*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_of_images*size_of_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;

        for (int i=0;i<size_of_images;++i){
            for (int j=0;j<size_of_images;++j){
                final  int val=intValues[pixel++];
                if(quantization_value==0){
                    byteBuffer.put((byte) ((val>>16)&0xFF));
                    byteBuffer.put((byte) ((val>>8)&0xFF));
                    byteBuffer.put((byte) (val&0xFF));
                }
                else {
                    byteBuffer.putFloat((((val >> 16) & 0xFF)));
                    byteBuffer.putFloat((((val >> 8) & 0xFF)));
                    byteBuffer.putFloat((((val) & 0xFF)));
                }
            }
        }
        return byteBuffer;
    }

}