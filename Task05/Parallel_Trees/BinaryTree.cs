using System;
using System.Linq;
using System.Threading;

namespace Parallel_Trees
{
    public abstract class BinaryTree<K, V> 
        where K : IComparable
        where V : struct
    {
        public class Node
        {
            internal K Key { get; set; }
            internal V Value { get; set; }

            internal Node? Parent;
            internal Node? LeftSon;
            internal Node? RightSon;

            internal readonly Mutex Mtx = new Mutex();

            internal void ChangeSon(bool isLeftSon, Node son)
            {
                if (isLeftSon)
                {
                    LeftSon = son;
                }
                else
                {
                    RightSon = son;
                }

                if (son != null)
                {
                    son.Parent = this;
                }
            }
            internal Node(K key, V value, Node parent)
            {
                Key = key;
                Value = value;
                Parent = parent;
            }
        }

        internal Node? Root;

        public abstract Node Insert(K key, V value);

        public abstract V? Find(K key);

        public abstract bool Remove(K key);
        
        public void PrintTree()
        {
            if (Root != null)
            {
                PrintTreeRec(Root, 1); 
            }
        } 

        private void PrintTreeRec(Node node, int shift)
        {
            if (node.RightSon != null)
            {
                PrintTreeRec(node.RightSon, shift + 1);
                foreach (var _ in Enumerable.Range(0, shift))
                {
                    Console.Write(" ");
                }
                Console.WriteLine("/");
            }

            foreach (var _ in Enumerable.Range(0, shift))
            {
                Console.Write(" ");
            }
            Console.WriteLine(node.Key);

            if (node.LeftSon != null)
            {
                foreach (var _ in Enumerable.Range(0, shift))
                {
                    Console.Write(" ");
                }
                Console.WriteLine("\\");
                PrintTreeRec(node.LeftSon, shift + 1);
            }
        }

        protected bool ChangeRootOptional(Node possibleRoot, Node newRoot)
        {
            if (possibleRoot != Root) return false;
            
            Root = newRoot;
            if (newRoot != null) Root.Parent = null;
            return true;
        }
    }
}