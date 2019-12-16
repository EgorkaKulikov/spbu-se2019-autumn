using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;

namespace ParallelTree
{
    class Node<T> : IComparable<Node<T>> where T : IComparable
    {
        public T Value { get; private set; }

        public Node<T> LeftChild { get; set; } = null;
        public Node<T> RightChild { get; set; } = null;

        public Mutex Locker { get; private set; }

        public Node(T value)
        {
            Value = value;
            Locker = new Mutex();
        }

        public void DisposeChilds()
        {
            RightChild?.DisposeChilds();
            LeftChild?.DisposeChilds();

            RightChild = null;
            LeftChild  = null;
        }

        public int CompareTo(Node<T> other)
        {
            if (other == null)
                return 1;

            return Value.CompareTo(other.Value);
        }
    }

    class Tree<T> : ICollection<T> where T : IComparable
    {
        public Node<T> Root { get; protected set; } = null;

        protected int count = 0;
        public int Count => count;

        public bool IsReadOnly => false;

        public Tree() { }

        public virtual void Add(T value)
        {
            if (Root == null)
            {
                Root = new Node<T>(value);
                return;
            }

            var current = Root;
            Node<T> parent = null;

            while (current != null)
            {
                parent = current;

                switch(value.CompareTo(current.Value))
                {
                    case -1:
                        current = current.LeftChild;
                        break;

                    case 1:
                        current = current.RightChild;
                        break;

                    case 0:
                        return;
                }
            }

            if (value.CompareTo(parent.Value) < 0)
                parent.LeftChild = new Node<T>(value);
            else
                parent.RightChild = new Node<T>(value);
            count++;
        }

        public virtual void Clear()
        {
            if (Root == null)
                return;

            Root.LeftChild = null;
            Root.LeftChild = null;
            Root = null;
        }

        public virtual bool Contains(T value)
        {
            var foundNode = FindNode(value, Root);

            if (foundNode == null)
                return false;
            else
                return true;
        }

        protected virtual Node<T> FindNode(T value, Node<T> current)
        {
            if (current == null)
                return null;

            switch (current.Value.CompareTo(value))
            {
                //current.Value equal value
                case 0:
                    return current;

                //current.Value greater
                case 1:
                    return FindNode(value, current.LeftChild);

                //current.Value lesser
                case -1:
                    return FindNode(value, current.RightChild);
            }
            return null;
        }

        public void CopyTo(T[] array, int arrayIndex)
        {
            var valueList = BuildTList();

            int maxIterationNumber = Math.Min(array.Length, valueList.Count);

            for (int i = 0; i < maxIterationNumber - arrayIndex; i++)
                array[i + arrayIndex] = valueList[i];
        }

        protected virtual List<T> BuildTList()
        {
            List<T> list = new List<T>();

            AddNodeToList(Root, list);

            return list;
        }

        protected virtual void AddNodeToList(Node<T> current, List<T> list)
        {
            if (current == null)
                return;

            list.Add(current.Value);

            if (current.LeftChild != null)
                AddNodeToList(current.LeftChild, list);

            if (current.RightChild != null)
                AddNodeToList(current.RightChild, list);
        }

        public bool Remove(T item)
        {
            throw new NotImplementedException();
        }

        public IEnumerator<T> GetEnumerator()
        {
            var array = BuildTList().ToArray();
            return new TreeEnumerator<T>(array);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    class TreeEnumerator<T> : IEnumerator<T> 
    {
        private T[] nodes;
        private int position;
        private T current;

        public T Current => current;
        object IEnumerator.Current => Current;

        public TreeEnumerator(T[] nodes)
        {
            this.nodes = nodes;
            position = -1;
            current = default(T);
        }

        public bool MoveNext()
        {
            if (++position >= nodes.Length)
            {
                return false;
            }
            else
            {
                current = nodes[position];
            }
            return true;
        }

        public void Reset()
        {
            position = -1;
        }

        public void Dispose() { }
    }
}
