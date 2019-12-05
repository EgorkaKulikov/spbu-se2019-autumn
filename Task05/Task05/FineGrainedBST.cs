using System;
using System.Threading;

namespace Task05
{
    public class FineGrainedBST
    {
        public class Node
        {
            public int data;
            public Mutex mutex;
            public Node parent, left, right;
            public bool IsLeftSon;

            public Node(int newData, Node newParent, bool isLeftSon)
            {
                mutex = new Mutex();
                data = newData;
                parent = newParent;
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
                mutex.WaitOne();
                mutex.ReleaseMutex();
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
        
        private Mutex rootLock = new Mutex();
        private Node root;

        public void print()
        {
            if (root != null)
            {
                if (root.parent != null)
                {
                    Console.WriteLine("!!!");    
                }
                
                root.print(0);
            }
        }
        
        public bool verify()
        {
            rootLock.WaitOne();
            rootLock.ReleaseMutex();
            if (root != null)
            {
                return (root.parent == null && root.verify());
            }

            return true;
        }

        public bool find(int value)
        {
            rootLock.WaitOne();
            Node curNode = root;
            curNode?.mutex.WaitOne();
            rootLock.ReleaseMutex();
            Node prev = null;
            while (curNode != null)
            {
                if (curNode.data == value)
                {
                    curNode.mutex.ReleaseMutex();
                    prev?.mutex.ReleaseMutex();
                    return true;
                }

                if (curNode.data < value)
                {
                    curNode.right?.mutex.WaitOne();
                    prev?.mutex.ReleaseMutex();
                    prev = curNode;
                    curNode = curNode.right;
                }
                else
                {
                    curNode.left?.mutex.WaitOne();
                    prev?.mutex.ReleaseMutex();
                    prev = curNode;
                    curNode = curNode.left;
                }
            }
            
            prev?.mutex.ReleaseMutex();
            return false;
        }

        public void add(int value)
        {
            rootLock.WaitOne();
            Node curNode = root;
            if (curNode == null)
            {
                root = new Node(value, null, true);
                rootLock.ReleaseMutex();
                return;
            }
            curNode?.mutex.WaitOne();
            rootLock.ReleaseMutex();
            Node prev = null;
            while (curNode != null)
            {
                if (curNode.data <= value)
                {
                    if (curNode.right == null)
                    {
                        curNode.right = new Node(value, curNode, false);
                        curNode.mutex.ReleaseMutex();
                        prev?.mutex.ReleaseMutex();
                        return;
                    }
                    
                    curNode.right?.mutex.WaitOne();
                    prev?.mutex.ReleaseMutex();
                    prev = curNode;
                    curNode = curNode.right;
                }
                else
                {
                    if (curNode.left == null)
                    {
                        curNode.left = new Node(value, curNode, true);
                        curNode.mutex.ReleaseMutex();
                        prev?.mutex.ReleaseMutex();
                        return;
                    }
                    
                    curNode.left?.mutex.WaitOne();
                    prev?.mutex.ReleaseMutex();
                    prev = curNode;
                    curNode = curNode.left;
                }
            }
        }

        public void delete(int value)
        {
            rootLock.WaitOne();
            Node curNode = root;
            curNode?.mutex.WaitOne(); 
            //rootLock.ReleaseMutex();
            var rootReleased = false;
            Node prev = null;
            while (curNode != null)
            {
                if (curNode.data == value)
                {
                    curNode.left?.mutex.WaitOne();
                    curNode.right?.mutex.WaitOne();
                    Node toReattach = curNode.left;
                    if (toReattach == null)
                    {
                        toReattach = curNode.right;
                    }
                    else
                    {
                        var toReattachRight = toReattach;
                        var isFirstIter = true;
                        while (toReattachRight.right != null)
                        {
                            var tmp = toReattachRight;
                            toReattachRight.right?.mutex.WaitOne();
                            toReattachRight = toReattachRight.right;
                            if (isFirstIter)
                            {
                                isFirstIter = false;
                            }
                            else
                            {
                                tmp.mutex.ReleaseMutex();
                            }
                        }

                        toReattachRight.right = curNode.right;
                        if (curNode.right != null)
                        {
                            curNode.right.parent = toReattachRight;
                        }
                        curNode.right?.mutex.ReleaseMutex();
                        if (!isFirstIter)
                        {
                            toReattachRight.mutex.ReleaseMutex();
                        }
                    }

                    if (curNode.parent != null) //no locking necessary as prev is locked
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

                        toReattach?.mutex.ReleaseMutex();
                    }
                    else
                    {
                        if (rootReleased)
                        {
                            rootLock.WaitOne();
                            root = toReattach;
                            rootLock.ReleaseMutex();
                        }
                        else
                        {
                            toReattach.parent = null;
                            root = toReattach;
                            rootLock.ReleaseMutex();
                        }
                        toReattach?.mutex.ReleaseMutex();
                    }
                    
                    prev?.mutex.ReleaseMutex();
                    return;
                }
                
                if (curNode.data < value)
                {
                    curNode.right?.mutex.WaitOne();
                    prev?.mutex.ReleaseMutex();
                    prev = curNode;
                    curNode = curNode.right;
                }
                else
                {
                    curNode.left?.mutex.WaitOne();
                    prev?.mutex.ReleaseMutex();
                    prev = curNode;
                    curNode = curNode.left;
                }

                if (!rootReleased)
                {
                    rootLock.ReleaseMutex();
                    rootReleased = true;
                }
            }
            
            prev?.mutex.ReleaseMutex();
        }
    }
}