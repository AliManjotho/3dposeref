using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using UnityEditor;
using UnityEngine;
using UnityEngine.UI;


public class KPFrames
{
    int ITERATIONS = 2;
    private int _sequenceCount;
    private List<Frame> _frames = new List<Frame>();

    //Properties
    public List<Frame> Frames { get { return this._frames; } }
    public int NumFrames { get { return this._frames.Count; } }

    public int SequenceCount { get { return this._sequenceCount; } }



    //Constructors
    public KPFrames(int sequenceCount) { this._sequenceCount = sequenceCount; }



    //Methods
    public void push(Vector4[] points)
    {
        if (points != null)
        {
            Frame newFrame = new Frame(points);

            if (this._frames.Count < this._sequenceCount)
            {
                this._frames.Add(newFrame);
            }
            else
            {
                List<Frame> _f = new List<Frame>();

                for (int i = 1; i < this._frames.Count; i++)
                    _f.Add(this._frames[i]);

                _f.Add(newFrame);

                this._frames = _f;
            }
        }
    }



    public Vector4[] getRefinedKeypoints(Vector4[] currentKeypoints)
    {
        if (this.Frames.Count > 0)
        {
            Vector4[] kps = new Vector4[Global.POSE_NUM_JOINTS];

            for (int joint = 0; joint < Global.POSE_NUM_JOINTS; joint++)
                kps[joint] = getMaxJoint(joint, currentKeypoints);

            return kps;
        }
        else
        {
            return currentKeypoints;
        }        
    }
    


    public void clear()
    {
        this._frames.Clear();
    }






    private Vector4 getMaxJoint(int joint, Vector4[] currentKeypoints)
    {
        List<Vector4> joints = new List<Vector4>();        
        Vector4 bestJoint = currentKeypoints[joint];


        for (int f = 0; f < this.Frames.Count; f++)
            joints.Add(this.Frames[f].Points[joint]);

        for (int i = 1; i < ITERATIONS; i++)
        {
            for (int j = 0; j < joints.Count; j++)
            {
                Vector4 p1 = bestJoint;
                Vector4 p2 = joints[j];

                bool p1_isEmpty = Helper.isPointEmpty(p1);
                bool p1_hasDepth = Helper.hadDepth(p1);
                bool p2_isEmpty = Helper.isPointEmpty(p2);
                bool p2_hasDepth = Helper.hadDepth(p2);

                if (!p1_isEmpty && !p2_isEmpty)
                {
                    //if (p1.w < p2.w)
                        //bestJoint = joints[j];

                    if (!p1_hasDepth && p2_hasDepth)
                        bestJoint.z = p2.z;
                }
                else if (p1_isEmpty && !p2_isEmpty)
                {
                    bestJoint = joints[j];
                }
            }
        }

        return bestJoint;
    }

   


    public void drawPoses()
    {
        float offset = 1.5f;

        Color[] colors = new Color[] { Color.red, Color.cyan, Color.yellow, Color.green, Color.magenta, Color.red, Color.cyan, Color.yellow, Color.green, Color.magenta, Color.red, Color.cyan, Color.yellow, Color.green, Color.magenta };

        
        for (int f = 0; f < this._frames.Count; f++)
        {
            for (int joint = 0; joint < Global.stickJoints.GetLength(0); joint++)
            {
                //Skeleton for Detected Pose

                Vector3 s = this._frames[f].Points[Global.stickJoints[joint, 0]];
                Vector3 e = this._frames[f].Points[Global.stickJoints[joint, 1]];

                if ((s.x != Global.EMPTY_VALUE && s.y != Global.EMPTY_VALUE && s.z != Global.EMPTY_VALUE) && (e.x != Global.EMPTY_VALUE && e.y != Global.EMPTY_VALUE && e.z != Global.EMPTY_VALUE))
                {
                    s.x = s.x + offset;
                    e.x = e.x + offset;

                    Helper.DrawLine(s, e, colors[f]);
                }
            }

            offset += 0.7f;
        }
    }


    override public string ToString()
    {
        string str = "Seq. Count = " + this.SequenceCount + "\n";
        str += "Frame Count = " + this.Frames.Count + "\n";
        
        for (int i = 0; i < this._frames.Count; i++)
            str += this._frames[i].ToString() + "\n";

        return str;
    }
}