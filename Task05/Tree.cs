using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;

namespace Task05
{
    abstract class Tree<K, V, NT> : IEnumerator, IEnumerable
        where K : IComparable<K>
        where V : struct
        where NT : Node<K, V, NT>
    {
        protected Mutex globalMutex = new Mutex();
        public NT overRoot;
        public NT root
        {
            get
            {
                return overRoot.parent;
            }
            set
            {
                overRoot.parent = value;
            }
        }

        protected abstract NT createRoot(K key, V value);

        public V? this[K key] {
            get {
                if (root == null)
                    return null;
                NT node = root, nextNode;
                while ((nextNode = node.nextNode(key)) != null)
                    node = nextNode;
                if (node != null && node.key.CompareTo(key) == 0)
                    return node.value;
                return null;
            }
            set {
                setWithInfo(key, value);
            }
        }

        protected abstract NodeStatus setWithInfo(K key, V? value);

        public void insert(params (K, V)[] elements)
        {
            foreach ((K, V) element in elements)
            {
                if (setWithInfo(element.Item1, element.Item2) == NodeStatus.NodeUpdate)
                    throw new Exception("Function insert don't update nodes in tree");
            }
        }

        public void erase(params K[] elements)
        {
            foreach (K element in elements)
            {
                if (setWithInfo(element, null) == NodeStatus.NodeNotFounded)
                    throw new Exception("Function erase can't delete absent node");
            }
        }

        public K[] keys {
            get {
                Stack<K> result = new Stack<K>();
                foreach (NT node in this) {
                    result.Push(node.key);
                }
                return result.ToArray();
            }
        }

        protected Stack<NT> path = new Stack<NT>();

        public bool MoveNext() {
            if (path.Count == 0) {
                if (root == null)
                    path = new Stack<NT>();
                else
                {
                    NT nodeOnLeftmostPath = root;
                    path.Push(root);
                    while (nodeOnLeftmostPath.left != null)
                    {
                        nodeOnLeftmostPath = nodeOnLeftmostPath.left;
                        path.Push(nodeOnLeftmostPath);
                    }
                }
            }
            else if (path.Peek().right != null)
            {
                NT node = path.Peek().right;
                while (node != null)
                {
                    path.Push(node);
                    node = node.left;
                }
            }
            else
            {
                NT node = path.Peek();
                while (node.type == SonType.RightSon)
                {
                    node = node.parent;
                    path.Pop();
                }
                path.Pop();
            }
            return path.Count != 0;
        }

        public object Current {
            get => path.Peek();
        }

        public void Reset() {
            path = new Stack<NT>();
        }

        public IEnumerator GetEnumerator() => this;
    }
}
