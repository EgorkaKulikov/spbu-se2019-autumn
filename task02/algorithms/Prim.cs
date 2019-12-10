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
            var chunkSize = edges.Count / numberOfThreads;
            var chunks = Enumerable.Range(0, numberOfThreads).Select((v, i) => i).Select((v, i) =>
            {
                if (i == numberOfThreads - 1)
                {
                    return Enumerable.Range(chunkSize * i, edges.Count - chunkSize * i);
                }
                return Enumerable.Range(chunkSize * i, chunkSize);
            }).ToArray();

            var usedEdges = Enumerable.Range(0, numberOfVertices).Select(_ => null as Edge).ToArray();
            var inMst = Enumerable.Range(0, numberOfVertices).Select(_ => false).ToArray();
            var processed = 0;

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
                    threads[i] = new Thread(chunk =>
                    {
                        foreach (var i in (IEnumerable<Int32>)chunk)
                        {
                            Int32 other;

                            if (target == edges[i].first)
                            {
                                other = edges[i].second;
                            }
                            else if (target == edges[i].second)
                            {
                                other = edges[i].first;
                            }
                            else
                            {
                                continue;
                            }

                            if (!inMst[other] && IsLess(edges[i], usedEdges[other]))
                            {
                                usedEdges[other] = edges[i];
                            }
                        }
                    });

                    threads[i].Start(chunks[i]);
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
