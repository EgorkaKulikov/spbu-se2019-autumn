using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Task_02
{
    public static class PrimaAlgo
    {
        internal static int PrimFindMstConsistent(List<Edge> graph, int countOfVertex)
        {
            var MST = new List<int>(graph.Count);
            var resWeight = 0;
            var notUsedE = new List<Edge>(graph);
            var usedV = new List<int>();
            var notUsedV = new List<int>();

            for (var i = 0; i < countOfVertex; i++)
            {
                notUsedV.Add(i);
            }

            var rand = new Random();
            usedV.Add(rand.Next(0, countOfVertex));
            notUsedV.RemoveAt(usedV[0]);
            while (notUsedV.Count > 0)
            {
                var minE = -1;

                for (var i = 0; i < notUsedE.Count; i++)
                {
                    if ((usedV.IndexOf(notUsedE[i].From) != -1) && (notUsedV.IndexOf(notUsedE[i].To) != -1) ||
                        (usedV.IndexOf(notUsedE[i].To) != -1) && (notUsedV.IndexOf(notUsedE[i].From) != -1))
                    {
                        if (minE != -1)
                        {
                            if (notUsedE[i].Weight < notUsedE[minE].Weight)
                                minE = i;
                        }
                        else
                            minE = i;
                    }
                }
                
                if (usedV.IndexOf(notUsedE[minE].From) != -1)
                {
                    usedV.Add(notUsedE[minE].To);
                    notUsedV.Remove(notUsedE[minE].To);
                }
                else
                {
                    usedV.Add(notUsedE[minE].From);
                    notUsedV.Remove(notUsedE[minE].From);
                }

                MST.Add(notUsedE[minE].Weight);
                resWeight += notUsedE[minE].Weight;
                notUsedE.RemoveAt(minE);
            }

            return resWeight;
        }

        internal static int PrimFindMstParallel(List<Edge> graph, int countOfVertex)
        {
            
        }
    }
}