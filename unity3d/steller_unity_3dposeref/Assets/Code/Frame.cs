using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

public class Frame
{
    private Vector4[] _points;
    public Vector4[] Points { get { return this._points; } set { this._points  = value; } }

    public Frame() { }

    public Frame(Vector4[] points) { this._points = (Vector4[])(points.Clone()); }

    
    override public string ToString()
    {
        string str = "";

        for (int i = 0; i < this.Points.Length; i++)
            if(this.Points[i] != null)
                str += this.Points[i] + " ";

        return str;
    }
}

