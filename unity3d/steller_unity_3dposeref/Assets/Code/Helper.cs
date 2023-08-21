using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using OpenPose;

public class Helper
{
    public static void DrawLine(Vector3 start, Vector3 end, Color color)
    {
        Debug.DrawLine(start, end, color);
    }


    public static Vector4[] getEmptyKeypoints()
    {
        Vector4[] keypoints = new Vector4[Global.POSE_NUM_JOINTS];

        for (int i = 0; i < Global.POSE_NUM_JOINTS; i++)
            keypoints[i] = new Vector4(Global.EMPTY_VALUE, Global.EMPTY_VALUE, Global.EMPTY_VALUE, 0f);

        return keypoints;
    }

    
    public static float[,] getEmptyRotations()
    {
        float[,] rotations = new float[1, 4];

        rotations[0, 0] = 0f;
        rotations[0, 1] = 0f;
        rotations[0, 2] = 0f;
        rotations[0, 3] = -1.0f;
                
        return rotations;
    }


    public static bool isFrameEmpty(Vector4[] keypoints)
    {
        if (keypoints == null)
            return true;

        bool isEmpty = false;

        for (int i = 0; i < keypoints.Length; i++)
             isEmpty = isEmpty || isPointEmpty(keypoints[i]);

        return isEmpty;
    }







    public static Vector4 getEmptyPoint()
    {
        return new Vector4(Global.EMPTY_VALUE, Global.EMPTY_VALUE, Global.EMPTY_VALUE, 0f);
    }


    public static bool isPointEmpty(float x, float y)
    {
        return (x == Global.EMPTY_VALUE && y == Global.EMPTY_VALUE) || (x == 0f && y == 0f);
    }

    public static bool isPointEmpty(Vector4 point)
    {
        return (point.x == Global.EMPTY_VALUE && point.y == Global.EMPTY_VALUE) || (point.x == 0f && point.y == 0f);
    }

    public static bool hadDepth(Vector4 point)
    {
        return !(point.z == 0f || point.z == Global.EMPTY_VALUE);
    }

    public static float getDistance(float x1, float y1, float x2, float y2)
    {
        return (float)Math.Sqrt(Math.Pow((x2 - x1), 2) + Math.Pow((y2 - y1), 2));
    }

    public static void copyList(ref List<Vector4> source, ref List<Vector4> destination)
    {
        destination.Clear();
        for (int i = 0; i < source.Count; i++)
            destination.Add(source[i]);
    }




    public static void printDebug(string fieldName, Vector4[] keypoints)
    {
        Text txtDebugField = GameObject.Find(fieldName).GetComponent<Text>();
        string str = "";

        if (txtDebugField != null && keypoints != null)
        {
            for (int i = 0; i < keypoints.Length; i++)
            {
                if(keypoints[i] != null)
                    str += keypoints[i].ToString() + "\n";
            }
            txtDebugField.text = str;
        }        
    }


    public static void printDebug(Vector4[] keypoints)
    {
        string str = "";

        if (keypoints != null)
        {
            for (int i = 0; i < keypoints.Length; i++)
            {
                if (keypoints[i] != null)
                    str += keypoints[i].ToString() + "\n";
            }
            Debug.Log(str);
        }
    }

}
