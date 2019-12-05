using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace Task05
{
    public class CoarseGrainedBST
    {
        public class Node
        {
            public int data;
            public Node parent, left, right;
            public bool IsLeftSon;

            public Node(int newData, Node newParent, bool isLeftSon)
            {
                data = newData;
                parent = newParent;
                left = null;
                right = null;
                IsLeftSon = isLeftSon;
            }

            public void print(int indent)
            {
                right?.print(indent + 2);

                for (var i = 0; i < indent; i++)
                {
                    Console.Write(' ');
                }

                if (left != null)
                {
                    if (left.parent.Equals(this) && left.IsLeftSon)
                    {
                        Console.Write("+");
                    }
                    else
                    {
                        Console.Write("-");
                    }
                }
                
                Console.Write(data);

                if (right != null)
                {
                    if (right.parent.Equals(this) && !right.IsLeftSon)
                    {
                        Console.Write("+");
                    }
                    else
                    {
                        Console.Write("-");
                    }
                }
                
                Console.WriteLine("");
                left?.print(indent + 2);
            }
            
            public bool verify()
            {
                bool result = true;
                if (parent != null)
                {
                    if (IsLeftSon)
                    {
                        result = parent.left != null && parent.left.Equals(this);
                    }
                    else
                    {
                        result = parent.right != null && parent.right.Equals(this);
                    }
                }

                if (left != null)
                {
                    result = result && left.verify();
                }

                if (right != null)
                {
                    result = result && right.verify();
                }

                return result;
            }
        }

        private Mutex mutex = new Mutex();
        private Node root;

        public void print()
        {
            root?.print(0);
        }

        public bool verify()
        {
            mutex.WaitOne();
            mutex.ReleaseMutex();
            if (root != null)
            {
                return (root.parent == null && root.verify());
            }

            return true;
        }

        public bool find(int value)
        {
            mutex.WaitOne();
            Node curNode = root;
            while (curNode != null)
            {

                if (curNode.data == value)
                {
                    mutex.ReleaseMutex();
                    return true;
                }

                if (curNode.data < value)
                {
                    curNode = curNode.right;
                }
                else
                {
                    curNode = curNode.left;
                }
            }

            mutex.ReleaseMutex();
            return false;
        }

        public void add(int value)
        {
            mutex.WaitOne();
            Node curNode = root;
            while (curNode != null)
            {
                if (curNode.data <= value)
                {
                    if (curNode.right == null)
                    {
                        curNode.right = new Node(value, curNode, false);
                        mutex.ReleaseMutex();
                        return;
                    }
                    
                    curNode = curNode.right;
                }
                else
                {
                    if (curNode.left == null)
                    {
                        curNode.left = new Node(value, curNode, true);
                        mutex.ReleaseMutex();
                        return;
                    }
                    
                    curNode = curNode.left;
                }
            }

            root = new Node(value, null, true);
            mutex.ReleaseMutex();
        }

        public void delete(int value)
        {
            mutex.WaitOne();
            Node curNode = root;
            while (curNode != null)
            {
                if (curNode.data == value)
                {
                    Node toReattach = curNode.left;
                    if (toReattach == null)
                    {
                        toReattach = curNode.right;
                    }
                    else
                    {
                        var toReattachRight = toReattach;
                        while (toReattachRight.right != null)
                        {
                            toReattachRight = toReattachRight.right;
                        }

                        toReattachRight.right = curNode.right;
                        if (curNode.right != null)
                        {
                            curNode.right.parent = toReattachRight;
                        }
                    }

                    if (curNode.parent != null)
                    {
                        if (curNode.IsLeftSon)
                        {
                            curNode.parent.left = toReattach;
                            if (toReattach != null)
                            {
                                toReattach.IsLeftSon = true;
                                toReattach.parent = curNode.parent;
                            }
                        }
                        else
                        {
                            curNode.parent.right = toReattach;
                            if (toReattach != null)
                            {
                                toReattach.IsLeftSon = false;
                                toReattach.parent = curNode.parent;
                            }
                        }
                    }
                    else
                    {
                        toReattach.parent = null;
                        root = toReattach;
                    }
                    
                    mutex.ReleaseMutex();
                    return;
                }
                
                if (curNode.data < value)
                {
                    curNode = curNode.right;
                }
                else
                {
                    curNode = curNode.left;
                }
            }

            mutex.ReleaseMutex();
        }
    }
}