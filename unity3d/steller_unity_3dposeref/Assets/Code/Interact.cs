using System;
using System.Collections;
using System.Net;
using System.Threading;
using System.Text;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using OpenPose;
using System.Collections.Generic;


public class Interact : MonoBehaviour
{
    [SerializeField] private float stabilizationIndex = 0.0f;

    public Vector4[] keypoints = new Vector4[Global.POSE_NUM_JOINTS];
    public float[,] rotations = new float[1, 4];

    private OPDatum datum;
    public bool body_set = true;  


    private void Start()
    {
        this.keypoints = Helper.getEmptyKeypoints();
        this.rotations = Helper.getEmptyRotations();

        OPWrapper.OPRegisterCallbacks();
        OPWrapper.OPEnableDebug(true);
        OPWrapper.OPEnableOutput(true);
        OPWrapper.OPEnableImageOutput(true);

        //UserConfigureOpenPose
        ////////////////////////////////////////////////////////////////////////////////
        ProducerType inputType = ProducerType.Webcam;
        string producerString = "-1";

        Vector2Int netResolution = new Vector2Int(-1, 368);
        Vector2Int handResolution = new Vector2Int(368, 368);
        Vector2Int faceResolution = new Vector2Int(368, 368);

        int maxPeople = 1;
        bool handEnabled = false;
        bool faceEnabled = false;

        OPWrapper.OPConfigurePose(PoseMode.Enabled, netResolution, null, OpenPose.ScaleMode.InputResolution, -1, 0, 1, 0.3f, OpenPose.RenderMode.Gpu, PoseModel.BODY_25, true, 0.6f, 0.7f, 0, null, HeatMapType.None, OpenPose.ScaleMode.UnsignedChar, false, Global.RENDER_THRESHOLD, maxPeople, false, -1.0, "", "", 0f);
        OPWrapper.OPConfigureHand(handEnabled, Detector.Body, handResolution, 1, 0.4f, OpenPose.RenderMode.None, 0.6f, 0.7f, 0.2f);
        OPWrapper.OPConfigureFace(faceEnabled, Detector.Body, faceResolution, OpenPose.RenderMode.None, 0.6f, 0.7f, 0.4f);
        OPWrapper.OPConfigureExtra(false, -1, false, -1, 0);
        OPWrapper.OPConfigureInput(inputType, producerString, 0, 1, ulong.MaxValue, false, false, 0, false, null, null, false, -1);
        OPWrapper.OPConfigureOutput(-1.0, "", DataFormat.Yml, "", "", 1, 1, "", "png", "", 30.0, false, "", "png", "", "", "", "", "8051");
        OPWrapper.OPConfigureGui(DisplayMode.NoDisplay, false, false);
        OPWrapper.OPConfigureDebugging(Priority.High, false, 1000);
        ////////////////////////////////////////////////////////////////////////////////


        OPWrapper.OPRun();
        
    }
    
    void Update()
    {        
        Texture2D outputTexture = new Texture2D(100, 100);
        Texture2D skeletonTexture = new Texture2D(100, 100);
        
        if (OPWrapper.OPGetOutput(out datum))
        {
            collectPoints(ref datum);
                        
            MultiArray<byte> outputData = datum.cvInputData;
            MultiArray<byte> skeletonData = datum.cvSkeletonOutputData;

            if (outputData != null || !outputData.Empty())
            {
                int height = outputData.GetSize(0), width = outputData.GetSize(1);
                outputTexture.Resize(width, height, TextureFormat.RGB24, false);
                outputTexture.LoadRawTextureData(outputData.ToArray());
                outputTexture.Apply();

                RawImage cameraFrame = GameObject.Find("CameraFrame").GetComponent<RawImage>();
                cameraFrame.texture = outputTexture;
            }
                    
            if (skeletonData != null || !skeletonData.Empty())
            {
                int height = skeletonData.GetSize(0), width = skeletonData.GetSize(1);
                skeletonTexture.Resize(width, height, TextureFormat.RGB24, false);
                skeletonTexture.LoadRawTextureData(skeletonData.ToArray());
                skeletonTexture.Apply();

                RawImage skeletonFrame = GameObject.Find("SkeletonFrame").GetComponent<RawImage>();
                skeletonFrame.texture = skeletonTexture;
            }
              
            body_set = false;   
        }
        else
        {
            //Debug.Log("No Points Detected!");
        }        
    }













   





    public void collectPoints(ref OPDatum datum)
    {
        if (datum.poseKeypoints3D != null)
        {
            for (int i = 0; i < Global.POSE_NUM_JOINTS; i++)
            {
                float newX = datum.poseKeypoints3D.Get(0, Global.poseUnityMapper[i], 0);
                float newY = datum.poseKeypoints3D.Get(0, Global.poseUnityMapper[i], 1);
                float newZ = datum.poseKeypoints3D.Get(0, Global.poseUnityMapper[i], 2);
                float newS = datum.poseKeypoints3D.Get(0, Global.poseUnityMapper[i], 3);

                if (this.keypoints != null && this.keypoints[i] != null && Helper.getDistance(newX, newY, this.keypoints[i].x, this.keypoints[i].y) > stabilizationIndex)
                    this.keypoints[i] = new Vector4(newX, newY, newZ, newS);
            }

            if (datum.rotations != null)
            {
                this.rotations[0, 0] = datum.rotations.Get(0, 0, 0);
                this.rotations[0, 1] = datum.rotations.Get(0, 0, 1);
                this.rotations[0, 2] = datum.rotations.Get(0, 0, 2);
                this.rotations[0, 3] = datum.rotations.Get(0, 0, 3);
            }
            else
            {
                this.rotations = Helper.getEmptyRotations();
            }
        }
        else
        {
            this.keypoints = Helper.getEmptyKeypoints();
            this.rotations = Helper.getEmptyRotations();
        }        
    }
}