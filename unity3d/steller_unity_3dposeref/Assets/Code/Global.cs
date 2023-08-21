using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public enum BodyType : byte
{
    FULL_BODY,
    UPPER_BODY
}

public enum OPJOINT : byte
{
    NOSE,
    NECK,
    RSHOULDER,
    RELBOW,
    RHAND,
    LSHOULDER,
    LELBOW,
    LHAND,
    MHIP,
    RHIP,
    RKNEE,
    RFOOT,
    LHIP,
    LKNEE,
    LFOOT,
    REYE,
    LEYE,
    REAR,
    LEAR,
    LBIGTOE,
    LSMALLTOE,
    LHEEL,
    RBIGTOE,
    RSMALLTOE,
    RHEEL,
    CHEST,
    TORSO
}

public enum UJOINT : byte
{
    NOSE,
    NECK,
    RSHOULDER,
    RELBOW,
    RHAND,
    LSHOULDER,
    LELBOW,
    LHAND,
    CHEST,
    TORSO,
    MHIP,
    RHIP,
    RKNEE,
    RFOOT,
    LHIP,
    LKNEE,
    LFOOT,
    REYE,
    LEYE
}


public class Global
{
    public static float EMPTY_VALUE = -99999.0f;
    public static float RENDER_THRESHOLD = 0.05f;

    public const short POSE_NUM_JOINTS = 17;
    public const short POSE_NUM_BONES = 16;

    public static float MIN_Z = 2.0f;
    public static float MAX_Z = 5.0f;

    public static int[] poseUnityMapper = { 0, 1, 2, 3, 4, 5, 6, 7, 25, 26, 8, 9, 10, 11, 12, 13, 14 };

    public static int[,] stickJoints = new int[,] { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 4 }, { 1, 5 }, { 5, 6 }, { 6, 7 }, { 1, 8 }, { 8, 9 }, { 9, 10 }, { 10, 11 }, { 11, 12 }, { 12, 13 }, { 10, 14 }, { 14, 15 }, { 15, 16 } };
}
