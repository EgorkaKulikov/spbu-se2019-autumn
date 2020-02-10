using System;
using System.Collections;
using System.Collections.Generic;

namespace Task05
{
    abstract public class BinaryTree<K, V, NT> : IEnumerable
        where K : IComparable<K>
        where V : struct
        where NT : BinaryNode<K, V>
    {
        public NT Root;

        public V? this[K key]
        {
            get
            {
                return Get(key);
            }
            set
            {
                SetWithInfo(key, value);
            }
        }

        public abstract V? Get(K key);

        public abstract NodeStatus SetWithInfo(K key, V? value);

        public void Insert(params (K, V)[] elements)
        {
            foreach ((K, V) element in elements)
            {
                if (SetWithInfo(element.Item1, element.Item2) == NodeStatus.NodeUpdate)
                    throw new Exception("Function insert don't update nodes in tree");
            }
        }

        public void Erase(params K[] elements)
        {
            foreach (K element in elements)
            {
                if (SetWithInfo(element, null) == NodeStatus.NodeNotFounded)
                    throw new Exception("Function erase can't delete absent node");
            }
        }

        public K[] Keys
        {
            get
            {
                Stack<K> resultStack = new Stack<K>();
                foreach (NT node in this)
                {
                    resultStack.Push(node.Key);
                }
                var resultArray = resultStack.ToArray();
                Array.Reverse(resultArray);
                return resultArray;
            }
        }

        public IEnumerator GetEnumerator() => new TreeEnumerator(Root);

        private class TreeEnumerator : IEnumerator
        {
            private readonly BinaryNode<K, V> Root;

            public TreeEnumerator(NT root)
            {
                Root = root;
            }

            private Stack<BinaryNode<K, V>> path = new Stack<BinaryNode<K, V>>();

            public bool MoveNext()
            {
                if (path.Count == 0)
                {
                    if (Root == null)
                        path = new Stack<BinaryNode<K, V>>();
                    else
                    {
                        var nodeOnLeftmostPath = Root;
                        path.Push(Root);
                        while (nodeOnLeftmostPath.Left != null)
                        {
                            nodeOnLeftmostPath = nodeOnLeftmostPath.Left;
                            path.Push(nodeOnLeftmostPath);
                        }
                    }
                }
                else if (path.Peek().Right != null)
                {
                    var node = path.Peek().Right;
                    while (node != null)
                    {
                        path.Push(node);
                        node = node.Left;
                    }
                }
                else
                {
                    var node = path.Peek();
                    while (node.Type == SonType.RightSon)
                    {
                        node = node.Parent;
                        path.Pop();
                    }
                    path.Pop();
                }
                return path.Count != 0;
            }

            public object Current => path.Peek();

            public void Reset()
            {
                path = new Stack<BinaryNode<K, V>>();
            }
        }
    }
}
