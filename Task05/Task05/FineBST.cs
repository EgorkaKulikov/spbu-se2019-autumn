using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;

namespace Task05
{
    public class FineBST
    {
        public class Node
        {
            public int value;
            public Node left, right;
            internal Mutex nodeMutex;

            public Node(int value)
            {
                nodeMutex = new Mutex();
                this.value = value;
                left = null;
                right = null;
            }

            public bool IsCorrect()
            {
                bool isReleased = nodeMutex.WaitOne(0);
                if (!isReleased)
                {
                    return false;
                }

                nodeMutex.ReleaseMutex();
                bool isCorrect = true;

                if (left != null)
                {
                    isCorrect = isCorrect && (left.value < value) && left.IsCorrect();
                }

                if (right != null)
                {
                    isCorrect = isCorrect && (value < right.value) && right.IsCorrect();
                }

                return isCorrect;
            }
        }
        public bool IsCorrect()
        {
            bool isReleased = rootMutex.WaitOne(0);
            if (!isReleased)
            {
                return false;
            }
            rootMutex.ReleaseMutex();
            if (root != null)
            {
                return root.IsCorrect();
            }

            return true;
        }

        private Node root;
        private readonly Mutex rootMutex = new Mutex();

        public void Insert(int value)
        {
            rootMutex.WaitOne();
            var current = root;

            if (root == null)
            {
                root = new Node(value);
                rootMutex.ReleaseMutex();
                return;
            }

            current.nodeMutex.WaitOne();
            rootMutex.ReleaseMutex();

            while(true)
            {
                if (value > current.value)
                {
                    if (current.right == null)
                    {
                        current.right = new Node(value);
                        current.nodeMutex.ReleaseMutex();
                        return;
                    }
                    current.right.nodeMutex.WaitOne();
                    current.nodeMutex.ReleaseMutex();
                    current = current.right;
                }
                else if (value < current.value)
                {
                    if (current.left == null)
                    {
                        current.left = new Node(value);
                        current.nodeMutex.ReleaseMutex();
                        return;
                    }
                    current.left.nodeMutex.WaitOne();
                    current.nodeMutex.ReleaseMutex();
                    current = current.left;
                }
                else
                {
                    current.nodeMutex.ReleaseMutex();
                    return;
                }
            }
        }

        public bool Find(int value)
        {
            rootMutex.WaitOne();
            var current = root;
            current?.nodeMutex.WaitOne();
            rootMutex.ReleaseMutex();
            while (current != null)
            {
                if (current.value == value)
                {
                    current.nodeMutex.ReleaseMutex();
                    return true;
                }

                if (value < current.value)
                {
                    current.left?.nodeMutex.WaitOne();
                    current.nodeMutex.ReleaseMutex();
                    current = current.left;
                }
                else
                {
                    current.right?.nodeMutex.WaitOne();
                    current.nodeMutex.ReleaseMutex();
                    current = current.right;
                }
            }
            return false;
        }
    }
}
