using System;
using System.Threading;
namespace Task05
{
    public class CoarseBST
    {
        public class Node
        {
            public int value;
            public Node left, right;

            public Node(int value)
            {
                this.value = value;
                left = null;
                right = null;
            }

            public bool IsCorrect()
            {
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
            return IsReleased() && (root == null || root.IsCorrect());
        }

        private Node root;
        private readonly Mutex treeMutex = new Mutex();

        private bool IsReleased()
        {
            var isReleased = treeMutex.WaitOne(0);
            if (!isReleased)
            {
                return false;
            }
            
            treeMutex.ReleaseMutex();
            return true;
        }

        
        public void Insert(int value)
        {
            treeMutex.WaitOne();
            Node current = root;

            if (root == null)
            {
                root = new Node(value);
                treeMutex.ReleaseMutex();
                return;
            }

            while (true)
            {
                if (value < current.value)
                {
                    if (current.left == null)
                    {
                        current.left = new Node(value);
                        treeMutex.ReleaseMutex();
                        return;
                    }
                    current = current.left;
                }
                else if (value > current.value)
                {
                    if (current.right == null)
                    {
                        current.right = new Node(value);
                        treeMutex.ReleaseMutex();
                        return;
                    }
                    current = current.right;
                }
                else
                {
                    treeMutex.ReleaseMutex();
                    return;
                }
            }
        }

        public bool Find(int value)
        {
            treeMutex.WaitOne();
            Node current = root;
 
            while (current != null)
            {
                if (current.value == value)
                {
                    treeMutex.ReleaseMutex();
                    return true;
                }
 
                if (value < current.value)
                {
                    current = current.left;
                }
                else
                {
                    current = current.right;
                }
            }

            treeMutex.ReleaseMutex();
            return false;
        }
    }
}
