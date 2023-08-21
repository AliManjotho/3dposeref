using System.Collections;
using UnityEditor;
using UnityEngine.UI;
using System.Collections.Generic;
using UnityEngine;
using System;
using RootMotion.FinalIK;

public class FinalIK : MonoBehaviour
{
    private int frameNum = 0;
    private KPFrames KP_Frames = new KPFrames(5);
    
    public FullBodyBipedIK ik;
    GameObject[] targetObjects = new GameObject[Global.POSE_NUM_JOINTS];
    GameObject targets;
    GameObject character;
    Camera mainCamera;
    
    private string[] JointNames = { "head", "neckLower", "rShldrBend", "rForearmBend", "rHand", "lShldrBend", "lForearmBend", "lHand", "chestUpper", "abdomenLower", "hip", "rThighBend", "rShin", "rFoot", "lThighBend", "lShin", "lFoot"};

    [SerializeField] private bool considerPreviousFrames = false;
    [SerializeField] public Interact interactor;
    [SerializeField] private BodyType BodyType = BodyType.FULL_BODY;
    [SerializeField] private float speed = 2.0f;
    [SerializeField] private float armBendWeight = 0f;
    [SerializeField] private float legBendWeight = 0f;
    [SerializeField, Range(10, 120)] private float FrameRate = 30.0f;

    private Vector4[] keypoints;
    private float[,] rotations;
    private float[] lerpFractions = { 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f };
    private float headLerpFraction = 0f;
    
    private List<Transform> BoneList = new List<Transform>();
    private float[] BoneDistance = new float[Global.POSE_NUM_BONES];
    private float Timer;




    void Start()
    {
        this.keypoints = Helper.getEmptyKeypoints();
        this.rotations = Helper.getEmptyRotations();

        character = GameObject.Find("steller");
        targets = new GameObject("Targets");
        mainCamera = GameObject.Find("MainCamera").GetComponent<Camera>();

        for (int i = 0; i < Global.POSE_NUM_JOINTS; i++)
        {
            targetObjects[i] = new GameObject(Enum.GetNames(typeof(UJOINT))[i] + "Target");
            targetObjects[i].transform.parent = targets.transform;
        }



        

        BoneList.Clear();

        for (int i = 0; i < JointNames.Length; i++)
        {
            Transform obj = GameObject.Find(JointNames[i]).transform;
            if (obj) BoneList.Add(obj);
        }

        for (int i = 0; i < Global.POSE_NUM_BONES; i++)
            BoneDistance[i] = Vector3.Distance(BoneList[Global.stickJoints[i, 0]].position, BoneList[Global.stickJoints[i, 1]].position);




        
        if (BodyType == BodyType.FULL_BODY)
        {
            //Body
            ik.solver.bodyEffector.target = targetObjects[(int)UJOINT.TORSO].transform;
            ik.solver.bodyEffector.positionWeight = 1.0f;


            //Right Arm
            ik.solver.rightHandEffector.target = targetObjects[(int)UJOINT.RHAND].transform;
            ik.solver.rightHandEffector.positionWeight = 1.0f;

            ik.solver.rightShoulderEffector.target = targetObjects[(int)UJOINT.RSHOULDER].transform;
            ik.solver.rightShoulderEffector.positionWeight = 1.0f;

            ik.solver.rightArmChain.bendConstraint.bendGoal = targetObjects[(int)UJOINT.RELBOW].transform;
            ik.solver.rightArmChain.bendConstraint.weight = armBendWeight;



            //Left Arm
            ik.solver.leftHandEffector.target = targetObjects[(int)UJOINT.LHAND].transform;
            ik.solver.leftHandEffector.positionWeight = 1.0f;

            ik.solver.leftShoulderEffector.target = targetObjects[(int)UJOINT.LSHOULDER].transform;
            ik.solver.leftShoulderEffector.positionWeight = 1.0f;

            ik.solver.leftArmChain.bendConstraint.bendGoal = targetObjects[(int)UJOINT.LELBOW].transform;
            ik.solver.leftArmChain.bendConstraint.weight = armBendWeight;



            //Right Leg
            ik.solver.rightFootEffector.target = targetObjects[(int)UJOINT.RFOOT].transform;
            ik.solver.rightFootEffector.positionWeight = 1.0f;

            ik.solver.rightThighEffector.target = targetObjects[(int)UJOINT.RHIP].transform;
            ik.solver.rightThighEffector.positionWeight = 1.0f;

            ik.solver.rightLegChain.bendConstraint.bendGoal = targetObjects[(int)UJOINT.RKNEE].transform;
            ik.solver.rightLegChain.bendConstraint.weight = legBendWeight;


            //Left Leg
            ik.solver.leftFootEffector.target = targetObjects[(int)UJOINT.LFOOT].transform;
            ik.solver.leftFootEffector.positionWeight = 1.0f;

            ik.solver.leftThighEffector.target = targetObjects[(int)UJOINT.LHIP].transform;
            ik.solver.leftThighEffector.positionWeight = 1.0f;

            ik.solver.leftLegChain.bendConstraint.bendGoal = targetObjects[(int)UJOINT.LKNEE].transform;
            ik.solver.leftLegChain.bendConstraint.weight = legBendWeight;
        }



        else if (BodyType == BodyType.UPPER_BODY)
        {
            mainCamera.transform.position = new Vector3(0f, 0f, 0f);

            //Body
            ik.solver.bodyEffector.target = targetObjects[(int)UJOINT.TORSO].transform;
            ik.solver.bodyEffector.positionWeight = 1.0f;


            //Right Arm
            ik.solver.rightHandEffector.target = targetObjects[(int)UJOINT.RHAND].transform;
            ik.solver.rightHandEffector.positionWeight = 1.0f;

            ik.solver.rightShoulderEffector.target = targetObjects[(int)UJOINT.RSHOULDER].transform;
            ik.solver.rightShoulderEffector.positionWeight = 1.0f;

            ik.solver.rightArmChain.bendConstraint.bendGoal = targetObjects[(int)UJOINT.RELBOW].transform;
            ik.solver.rightArmChain.bendConstraint.weight = armBendWeight;



            //Left Arm
            ik.solver.leftHandEffector.target = targetObjects[(int)UJOINT.LHAND].transform;
            ik.solver.leftHandEffector.positionWeight = 1.0f;

            ik.solver.leftShoulderEffector.target = targetObjects[(int)UJOINT.LSHOULDER].transform;
            ik.solver.leftShoulderEffector.positionWeight = 1.0f;

            ik.solver.leftArmChain.bendConstraint.bendGoal = targetObjects[(int)UJOINT.LELBOW].transform;
            ik.solver.leftArmChain.bendConstraint.weight = armBendWeight;

        }

        //Helper.printDebug(this.keypoints);
    }





    void LateUpdate()
    {
        Timer += Time.deltaTime;
        if (Timer > (1 / FrameRate))
        {
            Timer = 0;
            pointUpdate();

            if (considerPreviousFrames)
            {
                KP_Frames.push(this.keypoints);
                if (this.keypoints != null && frameNum % 15 == 0)
                {
                    Vector4[] bestKeypoints = KP_Frames.getRefinedKeypoints(this.keypoints);
                    if (bestKeypoints != null) this.keypoints = bestKeypoints;
                }

                //frameNum++;

            }

            setEffectors();
            rotateHead();
        }

        //setEffectors();

        if (considerPreviousFrames)
        {
            KP_Frames.drawPoses();
            frameNum++;
        }

        
        //drawSkeleton();

        //Helper.printDebug(this.keypoints);
    }


    void pointUpdate()
    {
        if (interactor.body_set == true)
            return;

        if (interactor.keypoints != null)
        {
            keypoints = new Vector4[Global.POSE_NUM_JOINTS];

            for (int i = 0; i < Global.POSE_NUM_JOINTS; i++)
            {
                if (interactor.keypoints[i] != null)
                {
                    if (Helper.isPointEmpty(interactor.keypoints[i]))
                        this.keypoints[i] = Helper.getEmptyPoint();
                    else
                        this.keypoints[i] = new Vector4(interactor.keypoints[i].x / 1000.0f, interactor.keypoints[i].y / 1000.0f, interactor.keypoints[i].z / 1000.0f, interactor.keypoints[i].w);
                }
            }
            
            this.rotations = interactor.rotations;

            interactor.body_set = true;
        }
        else
        {
            interactor.body_set = false;
        }
    }


    void rotateHead()
    {
        Transform head = GameObject.Find("neckUpper").transform;
        Quaternion currentRot = head.transform.rotation;


        if (this.rotations != null && this.rotations.Length > 0)
        {
            float rotX = 0.0f, rotY = 0.0f, rotZ = 0.0f;
            short direction = -1;

            if (this.rotations[0, 3] != Global.EMPTY_VALUE)
                direction = (short)this.rotations[0, 3];


            if (direction == 1)
            {
                //if (this.rotations[0, 0] != Global.EMPTY_VALUE)
                //rotX = (this.rotations[0, 0] + 115) * 5;

                if (this.rotations[0, 1] != Global.EMPTY_VALUE)
                    rotY = this.rotations[0, 1] + 180.0f;

                if (this.rotations[0, 2] != Global.EMPTY_VALUE)
                    rotZ = -this.rotations[0, 2];

                headLerpFraction += Time.deltaTime * speed;
                head.rotation = Quaternion.Slerp(currentRot, Quaternion.Euler(rotX, rotY, rotZ), headLerpFraction);
            }
            else if (direction == 2)
            {
                //if (this.rotations[0, 0] != Global.EMPTY_VALUE)
                //rotX = (this.rotations[0, 0] + 115) * 5;

                if (this.rotations[0, 1] != Global.EMPTY_VALUE)
                    rotY = this.rotations[0, 1] + 180.0f;

                if (this.rotations[0, 2] != Global.EMPTY_VALUE)
                    rotZ = this.rotations[0, 2];

                headLerpFraction += Time.deltaTime * speed;
                head.rotation = Quaternion.Slerp(currentRot, Quaternion.Euler(rotX, rotY, rotZ), headLerpFraction);
            }
            else if (direction == 3 || direction == 4)
            {
                if (this.rotations[0, 0] != Global.EMPTY_VALUE)
                    rotX = this.rotations[0, 0];

                headLerpFraction += Time.deltaTime * speed;
                head.rotation = Quaternion.Slerp(currentRot, Quaternion.Euler(rotX, 180.0f, rotZ), headLerpFraction);
            }
        }

    }

    void setEffectors()
    {
        //if (this.keypoints != null  && this.keypoints[(int)UJOINT.TORSO] != null && !Global.isPointEmpty(this.keypoints[(int)UJOINT.TORSO]) && this.keypoints[(int)UJOINT.TORSO].z > Global.MIN_Z)
        if (this.keypoints != null)
        {
            if (BodyType == BodyType.FULL_BODY)
            {
                for (int i = 0; i < Global.POSE_NUM_JOINTS; i++)
                {
                    Vector4 pos = keypoints[i];
                    Vector4 oldPos = Vector4.zero;
                    Vector4 newPos = Vector4.zero;

                    oldPos = targetObjects[i].transform.position;
                    newPos = (Helper.isPointEmpty(pos.x, pos.y) || pos.z == Global.EMPTY_VALUE) ? oldPos : pos;

                    if (lerpFractions[i] < 1)
                    {
                        lerpFractions[i] += Time.deltaTime * speed;
                        targetObjects[i].transform.position = Vector3.Lerp(oldPos, newPos, lerpFractions[i]);
                    }
                    else
                    {
                        targetObjects[i].transform.position = pos;
                    }
                }
            }
            else if (BodyType == BodyType.UPPER_BODY)
            {
                for (int i = 0; i < Global.POSE_NUM_JOINTS; i++)
                {
                    Vector4 pos;

                    if (i == 9)
                    {
                        pos = new Vector3(keypoints[1].x, keypoints[1].y - 0.5f, keypoints[1].z);
                    }
                    else
                    {
                        pos = keypoints[i];
                    }

                    Vector4 oldPos = Vector4.zero;
                    Vector4 newPos = Vector4.zero;

                    oldPos = targetObjects[i].transform.position;
                    newPos = (Helper.isPointEmpty(pos.x, pos.y) || pos.z == Global.EMPTY_VALUE) ? oldPos : pos;

                    if (lerpFractions[i] < 1)
                    {
                        lerpFractions[i] += Time.deltaTime * speed;
                        targetObjects[i].transform.position = Vector3.Lerp(oldPos, newPos, lerpFractions[i]);
                    }
                    else
                    {
                        targetObjects[i].transform.position = pos;
                    }
                }
            }
        }
    }

    
    













    void drawSkeleton()
    {
        if (this.keypoints != null)
        {
            for (int i = 0; i < Global.stickJoints.GetLength(0); i++)
            {
                //Skeleton for Detected Pose

                float offset = 0.0f;
                Vector3 s = keypoints[Global.stickJoints[i, 0]];
                Vector3 e = keypoints[Global.stickJoints[i, 1]];
                s[0] = s[0] + offset;
                e[0] = e[0] + offset;
                Helper.DrawLine(s, e, Color.green);

                //Skeleton for OP Character

                offset = -1.5f;
                s = keypoints[Global.stickJoints[i, 0]];
                e = keypoints[Global.stickJoints[i, 1]];
                s[0] = s[0] + offset;
                e[0] = e[0] + offset;
                Helper.DrawLine(s, e, Color.blue);
            }
        }
    }


    /*
    void OnDrawGizmosSelected()
    {
        if (Application.isPlaying)
        {
            Gizmos.color = Color.red;
            
            for (int i = 0; i < Global.stickJoints.GetLength(0); i++)
            {
                float offset = 1.5f;
                Vector3 s = BoneList[Global.stickJoints[i, 0]].position;
                Vector3 e = BoneList[Global.stickJoints[i, 1]].position;
                s[0] = s[0] + offset;
                e[0] = e[0] + offset;
                Helper.DrawLine(s, e, Color.red);
                Gizmos.DrawSphere(s, 0.035f);
                Gizmos.DrawSphere(e, 0.035f);
            }
        }           
    }
    */


}
