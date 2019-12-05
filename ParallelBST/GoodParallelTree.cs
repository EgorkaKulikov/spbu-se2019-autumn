using System;
using System.Collections.Generic;
using System.Threading;

namespace ParallelTree
{
    class GoodParallelTree<T> : Tree<T> where T : IComparable
    //Not really so good, but meh.
    {
        private Mutex rootLocker = new Mutex();

        public override void Add(T value)
        {
            rootLocker.WaitOne();

            if (Root == null)
            {   
                Root = new Node<T>(value);
                rootLocker.ReleaseMutex();
                return;
            }

            var current = Root;

            current?.Locker.WaitOne();
            rootLocker.ReleaseMutex();

            Node<T> parent = null;

            while (current != null)
            {
                parent?.Locker.ReleaseMutex();
                parent = current;

                switch (value.CompareTo(current.Value))
                {
                    case -1:
                        current.LeftChild?.Locker.WaitOne();
                        current = current.LeftChild;
                        break;

                    case 1:
                        current.RightChild?.Locker.WaitOne();
                        current = current.RightChild;
                        break;

                    case 0:
                        current?.Locker.ReleaseMutex();
                        return;
                }
            }

            if (value.CompareTo(parent.Value) < 0)
                parent.LeftChild = new Node<T>(value);
            else
                parent.RightChild = new Node<T>(value);

            parent?.Locker.ReleaseMutex();
            Interlocked.Increment(ref count);
        }

        public override bool Contains(T value)
        {
            Root?.Locker.WaitOne();
            return base.Contains(value);
        }

        protected override Node<T> FindNode(T value, Node<T> current)
        {
            if (current == null)
                return null;

            switch (current.Value.CompareTo(value))
            {
                //current.Value equal value
                case 0:
                    current.Locker.ReleaseMutex();
                    return current;

                //current.Value greater
                case 1:
                    current.LeftChild?.Locker.WaitOne();
                    current.Locker.ReleaseMutex();
                    return FindNode(value, current.LeftChild);

                //current.Value lesser
                case -1:
                    current.RightChild?.Locker.WaitOne();
                    current.Locker.ReleaseMutex();
                    return FindNode(value, current.RightChild);
            }
            return null;
        }

        public override void Clear()
        {
            rootLocker.WaitOne();
            if (Root == null)
            {
                rootLocker.ReleaseMutex();
                return;
            }

            Root.Locker.WaitOne();
            Root.DisposeChilds();
            Root.Locker.ReleaseMutex();

            Root = null;

            rootLocker.ReleaseMutex();
        }

        //Used in CopyTo(...) method
        protected override List<T> BuildTList()
        {
            Root?.Locker.WaitOne();
            return base.BuildTList();
        }

        //Used in BuildTList() method
        protected override void AddNodeToList(Node<T> current, List<T> list)
        {
            if (current == null)
                return;
            
            list.Add(current.Value);

            current.LeftChild?.Locker.WaitOne();
            current.RightChild?.Locker.WaitOne();

            current.Locker.ReleaseMutex();

            if (current.LeftChild != null)
            { 
                AddNodeToList(current.LeftChild, list);
            }

            if (current.RightChild != null)
            {
                AddNodeToList(current.RightChild, list);
            }
        }
    }
}
