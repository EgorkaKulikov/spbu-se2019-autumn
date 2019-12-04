using System;
using System.IO;
using System.Collections.Generic;

namespace Task02
{
  public partial class Program
  {
    public static int Main(string[] args)
    {
      string folderPath = @"example/";
      string filePath = string.Concat(folderPath, args[0]);

      if (!File.Exists(filePath))
      {
        Console.WriteLine($"There is no file named {args[1]}.");
        return -1;
      }

      int vNum;
      int[,] matrixFloyd;
      int[,] matrixPrim;
      Edge[] edgeList;

      using (StreamReader sr = File.OpenText(filePath))
      {
        string vBuf;
        vBuf = sr.ReadLine();
        var edgeListBuf = new List<Edge>();

        try
        {
          vNum = int.Parse(vBuf);

          if (0 >= vNum)
          {
            Console.WriteLine("Incorrect input!");
            Console.WriteLine($"Non-positive number of nodes: '{vNum}'");
            return -1;                    
          }
          
          matrixFloyd = new int[vNum, vNum];
          matrixPrim  = new int[vNum, vNum];
        }
        catch (FormatException)
        {
          Console.WriteLine("Incorrect input!");
          Console.WriteLine($"Number of nodes is not integer: '{vBuf}'");
          return -1;
        }

        string[] sBuf;
        int fIndex;
        int sIndex;
        int value;

        while (null != (vBuf = sr.ReadLine()))
        {
          try
          {
            sBuf = vBuf.Split(" ", 3);
            fIndex = int.Parse(sBuf[0]);
            sIndex = int.Parse(sBuf[1]);
            value  = int.Parse(sBuf[2]);

            if (0 > fIndex || 0 > sIndex || vNum <= fIndex || vNum <= sIndex)
            {
              Console.WriteLine("Incorrect input!");
              Console.WriteLine($"Index is out of range: '{vBuf}'");
              return -1;                    
            }
            
            if ( 0 == matrixFloyd[fIndex, sIndex])
            {
              matrixFloyd[sIndex, fIndex] = value;
              matrixFloyd[fIndex, sIndex] = value;
              
              matrixPrim[sIndex, fIndex]  = value;
              matrixPrim[fIndex, sIndex]  = value;
              
              edgeListBuf.Add(new Edge(fIndex, sIndex, value));
            }
            else
            {
              matrixFloyd[fIndex, sIndex] = Math.Min(value, matrixFloyd[fIndex, sIndex]);
              matrixFloyd[sIndex, fIndex] = matrixFloyd[fIndex, sIndex];
              
              matrixPrim[fIndex, sIndex] = matrixFloyd[fIndex, sIndex];
              matrixPrim[sIndex, fIndex] = matrixFloyd[fIndex, sIndex];
              
              for (int i = 0; i < edgeListBuf.Count; i++)
              {
                if (edgeListBuf[i].from == fIndex && edgeListBuf[i].to == sIndex
                || edgeListBuf[i].from == sIndex && edgeListBuf[i].to == fIndex)
                {
                  edgeListBuf[i] = new Edge(fIndex, sIndex, matrixFloyd[fIndex, sIndex]);
                }
              }
            }
          }
          catch (FormatException)
          {
            Console.WriteLine("Incorrect input!");
            Console.WriteLine($"An attribute of path is not integer: '{vBuf}'");
            return -1;
          }
        }

        edgeList = edgeListBuf.ToArray();

        sr.Close();
      }

      evalFloyd(matrixFloyd, vNum);

      int resultPrim = evalPrim(matrixPrim, vNum);
      int resultKruskal = evalKruskal(edgeList, vNum);

      string outPathFloyd = String.Concat(folderPath, "OutputFloyd");
      string outPathOstov = String.Concat(folderPath, "OutputSpanning");

      using (System.IO.StreamWriter file = new System.IO.StreamWriter(outPathOstov))
      {
        if (resultKruskal == resultPrim)
        {
          file.WriteLine($"{resultPrim}");
        }
        else
        {
          file.WriteLine($"{resultPrim} {resultKruskal}");
        }
    
        file.Close();
      }


      using (System.IO.StreamWriter file = new System.IO.StreamWriter(outPathFloyd))
      {
        for (int i = 0; i < vNum; i++)
        {
          for (int j = 0; j < vNum; j++)
          {
            file.Write($"{matrixFloyd[i, j].ToString("0    ")}");
          }

          file.WriteLine();
        }

        file.Close();
      }

      return 0;
    }
  }
}
