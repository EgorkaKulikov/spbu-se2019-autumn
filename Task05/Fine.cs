using System;
using System.ComponentModel.Design;
using System.IO;
using System.Threading;

namespace Task_05
{

    class BinaryTree
    {
        public class BinaryTreeNode
        {
            internal int key;
            internal BinaryTreeNode left;
            internal BinaryTreeNode right;
            internal Mutex mutex;

            public BinaryTreeNode()
            {
                left = null;
                right = null;
                mutex = new Mutex(true);
            }
        }
        
        internal BinaryTreeNode root;
        private int firstСame = 0;
        private Mutex firstCameMutex = new Mutex();

        public BinaryTree()
        {
            root = null;
        }

        public void Insert(int value, ref BinaryTreeNode thisNode)
        {
            if (thisNode == null && Interlocked.Increment(ref firstСame) == 1)
            {
                thisNode = new BinaryTreeNode();
                thisNode.key = value;
                thisNode.mutex.ReleaseMutex();
            }
            else
            {
                while (thisNode == null)
                {
                }

                thisNode.mutex.WaitOne();
                if (value < thisNode.key)
                {
                    thisNode.mutex.ReleaseMutex();
                    if (thisNode.left == null)
                    {
                        firstCameMutex.WaitOne();
                        firstСame = 0;
                        Insert(value, ref thisNode.left);
                        firstCameMutex.ReleaseMutex();
                    }
                    else Insert(value, ref thisNode.left);
                }
                else
                {
                    thisNode.mutex.ReleaseMutex();
                    if (thisNode.right == null)
                    {
                        firstCameMutex.WaitOne();
                        firstСame = 0;
                        Insert(value, ref thisNode.right);
                        firstCameMutex.ReleaseMutex();
                    }
                    else Insert(value, ref thisNode.right);

                }
            }
        }

        public void Remove(int value, ref BinaryTreeNode thisNode)
        {
            if (thisNode != null)
            {
                thisNode.mutex.WaitOne();
                if (value < thisNode.key)
                {
                    thisNode.mutex.ReleaseMutex();
                    Remove(value, ref thisNode.left);
                }
                else if (value > thisNode.key)
                {
                    thisNode.mutex.ReleaseMutex();
                    Remove(value, ref thisNode.right);
                }
                else if (thisNode.right != null && thisNode.left != null)
                {
                    thisNode.right.mutex.WaitOne();
                    thisNode.key = Min(thisNode.right).key;
                    thisNode.right.mutex.ReleaseMutex();
                    thisNode.mutex.ReleaseMutex();
                    Remove(thisNode.key, ref thisNode.right);
                }
                else if (thisNode.right != null)
                {
                    BinaryTreeNode parent = thisNode;
                    thisNode.right.mutex = thisNode.mutex;
                    thisNode = thisNode.right;
                    parent.mutex.ReleaseMutex();
                }
                else if (thisNode.left != null)
                {
                    BinaryTreeNode parent = thisNode;
                    thisNode.left.mutex = thisNode.mutex;
                    thisNode = thisNode.left;
                    parent.mutex.ReleaseMutex();
                }
                else thisNode = null;
            }
        }

        private BinaryTreeNode Min(BinaryTreeNode node)
        {
            node.mutex.WaitOne();
            if (node.left == null)
            {
                node.mutex.ReleaseMutex();
                return node;
            }
            else
            {
                node.mutex.ReleaseMutex();
                return Min(node.left);
            }
        }

        public BinaryTreeNode Find(int value, ref BinaryTreeNode thisNode)
        {
            if (thisNode == null || value == thisNode.key) return thisNode;
            else
            {
                thisNode.mutex.WaitOne();
                if (value < thisNode.key)
                {
                    thisNode.mutex.ReleaseMutex();
                    BinaryTreeNode result = Find(value, ref thisNode.left);
                    return result;
                }
                else
                {
                    thisNode.mutex.ReleaseMutex();
                    BinaryTreeNode result = Find(value, ref thisNode.right);
                    return result;
                }
            }
        }

        public void Dfs()
        {
            if (root != null)
            {
                Console.WriteLine(root.key);
                BinaryTreeNode parent = root;
                root = root.left;
                Dfs();
                root = parent.right;
                Dfs();
                root = parent;
            }
        }

        public bool CheckMutexes()
        {
            if (root != null)
            {
                if (!root.mutex.WaitOne(0)) return false;
                root.mutex.ReleaseMutex();
                BinaryTreeNode parent = root;
                root = root.left;
                if (!CheckMutexes())
                {
                    root = parent;
                    return false;
                }
                else
                {
                    root = parent.right;
                    bool result = CheckMutexes();
                    root = parent;
                    return result;
                }
            }
            else return true;
        }
    }

    internal class Program
    {
        private static StreamReader file;

        private static void Input(ref int[] array, int count)
        {
            for (int i = 0; i < count; i++)
            {
                array[i] = Convert.ToInt32(file.ReadLine());
            }
        }

        public static void Main(string[] args)
        {
            if (args.Length > 0)
            {
                using (file = File.OpenText(args[0]))
                {
                    int insertCount = Convert.ToInt32(file.ReadLine());
                    int removeCount = Convert.ToInt32(file.ReadLine());
                    int findCount = Convert.ToInt32(file.ReadLine());
                    int[] insertKeys = new int [insertCount];
                    int[] removeKeys = new int [removeCount];
                    int[] findKeys = new int [findCount];
                    Input(ref insertKeys, insertCount);
                    Input(ref removeKeys, removeCount);
                    Input(ref findKeys, findCount);

                    BinaryTree tree = new BinaryTree();
                    int completed = 0;
                    ManualResetEvent allDone = new ManualResetEvent(initialState: false);
                    for (int i = 0; i < insertCount; i++)
                    {
                        var th = new Thread(id =>
                        {
                            tree.Insert(insertKeys[(int) id], ref tree.root);
                            if (Interlocked.Increment(ref completed) == insertCount)
                            {
                                allDone.Set();
                            }
                        });
                        th.Start(i);
                    }

                    allDone.WaitOne();
                    //tree.Dfs();
                    completed = 0;
                    allDone = new ManualResetEvent(initialState: false);
                    if (removeCount == 0) allDone.Set();
                    for (int i = 0; i < removeCount; i++)
                    {
                        Thread th = new Thread(id =>
                        {
                            tree.Remove(removeKeys[(int) id], ref tree.root);
                            if (Interlocked.Increment(ref completed) == removeCount)
                            {
                                allDone.Set();
                            }
                        });
                        th.Start(i);
                    }

                    allDone.WaitOne();
                    /*Console.WriteLine();
                    tree.Dfs();
                    Console.WriteLine();*/
                    completed = 0;
                    allDone = new ManualResetEvent(initialState: false);
                    if (findCount == 0) allDone.Set();
                    for (int i = 0; i < findCount; i++)
                    {
                        Thread th = new Thread(id =>
                        {
                            tree.Find(findKeys[(int) id], ref tree.root);
                            if (Interlocked.Increment(ref completed) == findCount)
                            {
                                allDone.Set();
                            }
                        });
                        th.Start(i);
                    }
                    allDone.WaitOne();
                    if (!tree.CheckMutexes()) Console.WriteLine("Some mutexes aren't released!");
                }
            }
            else Console.WriteLine("You didn't provide a filename");
        }
    }
}