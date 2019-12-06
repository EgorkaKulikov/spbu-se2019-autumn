using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace Task02
{
    static class Prim
    {
        public static IEnumerable<Edge> BuildMst(Int32 numberOfVertices, List<Edge> edges)
        {
            var numberOfThreads = Environment.ProcessorCount;
            var threads = new Thread[numberOfThreads];
            var chunksSize = edges.Count / numberOfThreads;
            var chunksBegins = new Int32[numberOfThreads];
            var chunksEnds = new Int32[numberOfThreads];

            for (Int32 i = 0; i < numberOfThreads; i++)
            {
                chunksBegins[i] = chunksSize * i;
                chunksEnds[i] = chunksBegins[i] + chunksSize;
            }
            chunksEnds[numberOfThreads - 1] = edges.Count;

            var usedEdges = new Edge[numberOfVertices];
            var inMst = new Boolean[numberOfVertices];
            var processed = 0;

            for (Int32 i = 0; i < numberOfVertices; i++)
            {
                usedEdges[i] = null;
                inMst[i] = false;
            }

            static Boolean IsLess(Edge edge1, Edge edge2)
            {
                return edge2 == null || edge1.weight < edge2.weight;
            }

            while (processed != numberOfVertices)
            {
                var target = 0;

                for (Int32 i = 0; i < numberOfVertices; i++)
                {
                    if (!inMst[i])
                    {
                        target = i;
                        break;
                    }
                }

                for (Int32 i = target + 1; i < numberOfVertices; i++)
                {
                    if (!inMst[i] && usedEdges[i] != null && IsLess(usedEdges[i], usedEdges[target]))
                    {
                        target = i;
                    }
                }

                inMst[target] = true;

                for (Int32 i = 0; i < numberOfThreads; i++)
                {
                    threads[i] = new Thread(index =>
                    {
                        for (Int32 i = chunksBegins[(int)index]; i < chunksEnds[(int)index]; i++)
                        {
                            var other = edges[i].OtherThan(target);

                            if (other < 0)
                            {
                                continue;
                            }

                            if (!inMst[other] && IsLess(edges[i], usedEdges[other]))
                            {
                                usedEdges[other] = edges[i];
                            }
                        }
                    });

                    threads[i].Start(i);
                }

                for (Int32 i = 0; i < numberOfThreads; i++)
                {
                    threads[i].Join();
                }

                processed++;
            }

            return usedEdges.Where(edge => edge != null);
        }
    }
}