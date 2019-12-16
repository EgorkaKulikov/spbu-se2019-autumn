using System;
using System.Threading;

namespace Parallel_Trees
{
    public class CoarseGrainedBinaryTree<K, V> : BinaryTree<K, V>
        where K : IComparable
        where V : struct
    {
        private readonly Mutex _mutex = new Mutex();

        public override Node Insert(K key, V value)
        {
            _mutex.WaitOne();
            try
            {
                if (Root == null)
                {
                    Root = new Node(key, value, null);
                    return Root;
                }

                Node currNode = Root;

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
                _mutex.ReleaseMutex();
            }
        }

        public override V? Find(K key)
        {
            _mutex.WaitOne();
            try
            {
                var res = StandardFind(key, node => (node, node != null));

                return res.Item1?.Value;
            }
            finally
            {
                _mutex.ReleaseMutex();
            }
        }

        private (Node, bool) StandardFind(K key, Func<Node, (Node, bool)> func)
        {
            if (Root == null)
            {
                return (null, false);
            }

            var currNode = Root;
            while (currNode != null)
            {
                switch (key.CompareTo(currNode.Key))
                {
                    case 1:
                    {
                        currNode = currNode.RightSon;

                        continue;
                    }

                    case -1:
                    {
                        currNode = currNode.LeftSon;

                        continue;
                    }

                    case 0:
                    {
                        return func(currNode);
                    }
                }
            }

            return (null, false);
        }

        public override bool Remove(K key)
        {
            _mutex.WaitOne();
            try
            {
                var res = StandardFind(key, ReplaceDeletingNode);

                return res.Item2;
            }
            finally
            {
                _mutex.ReleaseMutex();
            }
        }

        private (Node, bool) ReplaceDeletingNode(Node currNode)
        {
            var parent = currNode.Parent;
            var leftS = currNode.LeftSon;
            var rightS = currNode.RightSon;

            var isCurrLeftSon = parent?.LeftSon == currNode;

            if (leftS == null && rightS?.LeftSon == null)
            {
                if (!ChangeRootOptional(currNode, rightS))
                {
                    parent?.ChangeSon(isCurrLeftSon, rightS);
                }

                return (rightS, true);
            }

            if (rightS == null && leftS.RightSon == null)
            {
                if (!ChangeRootOptional(currNode, leftS))
                {
                    parent?.ChangeSon(isCurrLeftSon, leftS);
                }

                return (leftS, true);
            }

            if (rightS?.LeftSon == null && leftS?.RightSon == null)
            {
                if (!ChangeRootOptional(currNode, leftS))
                {
                    parent?.ChangeSon(isCurrLeftSon, leftS);
                }

                leftS?.ChangeSon(false, rightS);

                return (leftS, true);
            }

            var rightD = rightS;
            var leftD = leftS;

            while (rightD?.LeftSon != null && leftD?.RightSon != null)
            {
                rightD = rightD.LeftSon;
                leftD = leftD.RightSon;
            }

            var isDeeperRight = rightD?.LeftSon != null;

            if (isDeeperRight)
            {
                while (rightD.LeftSon != null)
                {
                    rightD = rightD.LeftSon;
                }

                rightD.Parent.ChangeSon(true, rightD.RightSon);

                currNode.Key = rightD.Key;
                currNode.Value = rightD.Value;
            }
            else
            {
                while (leftD.RightSon != null)
                {
                    leftD = leftD.RightSon;
                }

                leftD?.Parent.ChangeSon(false, leftD.LeftSon);

                currNode.Key = leftD.Key;
                currNode.Value = leftD.Value;
            }

            return (currNode, true);
        }
    }
}