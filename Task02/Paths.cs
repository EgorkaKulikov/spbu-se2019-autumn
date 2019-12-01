using System;
using System.IO;

namespace Graphs
{
    public static class Paths
    {
        private static readonly string PathFolder = Environment.CurrentDirectory;

        private static readonly DirectoryInfo Dir = new DirectoryInfo(PathFolder);
        private static readonly string NewPath = Dir?.Parent?.Parent?.Parent?.ToString() ?? "";
        
        public static readonly string PathInitGraph = NewPath + @"\init_graph.txt";
        public static readonly string PathResFl = NewPath + @"\res_floyd.txt";
        public static readonly string PathResPr = NewPath + @"\res_prim.txt";
        public static readonly string PathResKr = NewPath + @"\res_kruskal.txt";
    }
}