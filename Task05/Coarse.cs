using System;
using System.IO;
using System.Threading;


namespace Task05
{
    class BinaryTree
    {
        public class BinaryTreeNode
        {
            public int key;
            public BinaryTreeNode left;
            public BinaryTreeNode right;

            public BinaryTreeNode()
            {
                left = null;
                right = null;
            }
        }
        
        public BinaryTreeNode root;
        private Mutex mInsert = new Mutex();
        private Mutex mRemove = new Mutex();
        private Mutex mFind = new Mutex();

        public BinaryTree()
        {
            root = null;
        } 

        public void Insert(int value)
        {
            mInsert.WaitOne();
            if (root == null)
            {
                root = new BinaryTreeNode();
                root.key = value;
            }
            else
            {
                BinaryTreeNode parent = root;
                if (value < root.key)
                {
                    root = root.left;
                    Insert(value);
                    parent.left = root;
                    root = parent;
                }
                else
                {
                    root = root.right;
                    Insert(value);
                    parent.right = root;
                    root = parent;
                }
            }
            mInsert.ReleaseMutex();
        }

        public void Remove(int value)
        {
            mRemove.WaitOne();
            if (root != null)
            {
                if (value < root.key)
                {
                    BinaryTreeNode parent = root;
                    root = root.left;
                    Remove(value);
                    parent.left = root;
                    root = parent;
                }
                else if (value > root.key)
                {
                    BinaryTreeNode parent = root;
                    root = root.right;
                    Remove(value);
                    parent.right = root;
                    root = parent;
                }
                else if (root.left != null && root.right != null)
                {
                    BinaryTreeNode parent = root;
                    parent.key = Min(root.right).key;
                    root = root.right;
                    Remove(parent.key);
                    parent.right = root;
                    root = parent;
                }
                else if (root.left != null) root = root.left;
                else if (root.right != null) root = root.right;
                else root = null;
            }
            mRemove.ReleaseMutex();
        }

        public BinaryTreeNode Min(BinaryTreeNode node)
        {
            if (node.left == null) return node;
            else return Min(node.left);
        }

        public BinaryTreeNode Find(int value)
        {
            mFind.WaitOne();
            if (root == null || value == root.key)
            {
                mFind.ReleaseMutex();
                return root;
            }
            else
            {
                BinaryTreeNode parent = root;
                if (value < root.key)
                {
                    root = root.left;
                    BinaryTreeNode result = Find(value);
                    root = parent;
                    mFind.ReleaseMutex();
                    return result;
                }
                else
                {
                    root = root.right;
                    BinaryTreeNode result = Find(value);
                    root = parent;
                    mFind.ReleaseMutex();
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
                    int insertCount = Convert.ToInt32(Console.ReadLine());
                    int removeCount = Convert.ToInt32(Console.ReadLine());
                    int findCount = Convert.ToInt32(Console.ReadLine());
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
                            tree.Insert(insertKeys[(int) id]);
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
                            tree.Remove(removeKeys[(int) id]);
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
                            tree.Find(findKeys[(int) id]);
                            if (Interlocked.Increment(ref completed) == findCount)
                            {
                                allDone.Set();
                            }
                        });
                        th.Start(i);
                    }

                    allDone.WaitOne();
                }
            }
        }
    }
}