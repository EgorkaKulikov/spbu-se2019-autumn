using System;
using System.Threading;

namespace Parallel_Trees
{
    public class FineGrainedBinaryTree<K, V> : BinaryTree<K, V>
        where K : IComparable
        where V : struct
    {
        private readonly Mutex _rootMutex = new Mutex();
        
        public override Node Insert(K key, V value)
        {
            _rootMutex.WaitOne();
            try
            {
                if (Root == null)
                {
                    Root = new Node(key, value, null);
                    return Root;
                }
            }
            finally
            {
                _rootMutex.ReleaseMutex();
            }


            Node currNode = Root;
            currNode.Mtx.WaitOne();

            try
            {
                while (true)
                {
                    switch (key.CompareTo(currNode.Key))
                    {
                        case 1:
                        {
                            if (currNode.RightSon == null)
                            {
                                currNode.RightSon = new Node(key, value, currNode);
                                return currNode.RightSon;
                            }

                            currNode.RightSon.Mtx.WaitOne();
                            currNode.Mtx.ReleaseMutex();

                            currNode = currNode.RightSon;

                            continue;
                        }

                        case -1:
                        {
                            if (currNode.LeftSon == null)
                            {
                                currNode.LeftSon = new Node(key, value, currNode);
                                return currNode.LeftSon;
                            }

                            currNode.LeftSon.Mtx.WaitOne();
                            currNode.Mtx.ReleaseMutex();

                            currNode = currNode.LeftSon;

                            continue;
                        }
                        case 0:
                        {
                            currNode.Value = value;
                            return currNode;
                        }
                    }
                }
            }
            finally
            {
                currNode.Mtx.ReleaseMutex();
            }
        }

        public override V? Find(K key)
        {
            var res = StandardFind(key, node => (node, node != null));

            return res.Item1?.Value;
        }

        private (Node, bool) StandardFind(K key, Func<Node, (Node, bool)> func)
        {
            _rootMutex.WaitOne();
            try
            {
                if (Root == null)
                {
                    return (null, false);
                }
                else
                {
                    Root.Mtx.WaitOne();
                }
            }
            finally
            {
                _rootMutex.ReleaseMutex();
            }

            var currNode = Root;

            try
            {
                while (currNode != null)
                {
                    switch (key.CompareTo(currNode.Key))
                    {
                        case 1:
                        {
                            currNode.RightSon?.Mtx.WaitOne();
                            currNode.Mtx.ReleaseMutex();
                            
                            currNode = currNode.RightSon;

                            continue;
                        }

                        case -1:
                        {
                            currNode.LeftSon?.Mtx.WaitOne();
                            currNode.Mtx.ReleaseMutex();
                            
                            currNode = currNode.LeftSon;

                            continue;
                        }

                        case 0:
                        {
                            return func(currNode);
                        }
                    }
                }
            }
            finally
            {
                currNode?.Mtx.ReleaseMutex();
            }

            return (null, false);
        }

        public override bool Remove(K key)
        {
            var res = StandardFind(key, ReplaceDeletingNode);

            return res.Item2;
        }

        private (Node, bool) ReplaceDeletingNode(Node currNode)
        {
            var parent = currNode.Parent;
            var leftS = currNode.LeftSon;
            var rightS = currNode.RightSon;

            leftS?.Mtx.WaitOne();
            rightS?.Mtx.WaitOne();

            var isCurrLeftSon = parent?.LeftSon == currNode;

            if (leftS == null && rightS?.LeftSon == null)
            {
                try
                {
                    parent?.Mtx.WaitOne();

                    if (!ChangeRootOptional(currNode, rightS))
                    {
                        parent?.ChangeSon(isCurrLeftSon, rightS);
                    }

                    return (rightS, true);
                }
                finally
                {
                    rightS?.Mtx.ReleaseMutex();
                    parent?.Mtx.WaitOne();
                }
            }

            if (rightS == null && leftS.RightSon == null)
            {
                try
                {
                    parent?.Mtx.WaitOne();
                    
                    if (!ChangeRootOptional(currNode, leftS))
                    {
                        parent?.ChangeSon(isCurrLeftSon, leftS);
                    }

                    return (leftS, true);
                } 
                finally
                { 
                    leftS.Mtx.ReleaseMutex();
                    parent?.Mtx.WaitOne();
                }
            }

            if (rightS?.LeftSon == null && leftS?.RightSon == null)
            {
                try
                {
                    if (!ChangeRootOptional(currNode, leftS))
                    {
                        parent?.ChangeSon(isCurrLeftSon, leftS);
                    }

                    leftS?.ChangeSon(false, rightS);

                    return (leftS, true);
                }
                finally
                {
                    rightS?.Mtx.ReleaseMutex(); 
                    leftS?.Mtx.ReleaseMutex();
                }
            }

            var rightD = rightS;
            var leftD = leftS;

            while (rightD?.LeftSon != null && leftD?.RightSon != null)
            {
                rightD.LeftSon?.Mtx.WaitOne();
                leftD.RightSon?.Mtx.WaitOne();
                
                rightD?.Mtx.ReleaseMutex();
                leftD?.Mtx.ReleaseMutex();
                
                rightD = rightD.LeftSon;
                leftD = leftD.RightSon;
            }

            var isDeeperRight = rightD?.LeftSon != null;

            if (isDeeperRight)
            {
                leftD?.Mtx.ReleaseMutex();

                while (rightD.LeftSon != null)
                {
                    rightD.LeftSon.Mtx.WaitOne();
                    rightD.Mtx.ReleaseMutex();

                    rightD = rightD.LeftSon;
                }
                
                rightD?.RightSon?.Mtx.WaitOne();

                rightD.Parent.ChangeSon(true, rightD.RightSon);

                currNode.Key = rightD.Key;
                currNode.Value = rightD.Value;
                
                rightD?.RightSon?.Mtx.ReleaseMutex();
                rightD.Mtx.ReleaseMutex();
            }
            else
            {
                rightD?.Mtx.ReleaseMutex();

                while (leftD.RightSon != null)
                {
                    leftD.RightSon.Mtx.WaitOne();
                    leftD.Mtx.ReleaseMutex();

                    leftD = leftD.RightSon;
                }

                leftD?.LeftSon?.Mtx.WaitOne();

                leftD?.Parent.ChangeSon(false, leftD.LeftSon);

                currNode.Key = leftD.Key;
                currNode.Value = leftD.Value;
                
                leftD?.LeftSon?.Mtx.ReleaseMutex();
                leftD.Mtx.ReleaseMutex();
            }

            return (currNode, true);
        }
    }
}