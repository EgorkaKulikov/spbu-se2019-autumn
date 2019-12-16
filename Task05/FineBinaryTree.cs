
using System;
using System.Collections.Generic;
using System.Threading;

namespace Task05
{
    internal class FineBstNode<TK, TV> where TK : IComparable<TK>
    {
        internal TK Key;
        internal TV Value;
        internal FineBstNode<TK, TV> Parent;
        internal FineBstNode<TK, TV> Left;
        internal FineBstNode<TK, TV> Right;
        internal Mutex FineMutex = new Mutex();
        internal FineBstNode(TK key, TV value)
        {
            Key = key;
            Value = value;
        }
        internal FineBstNode(TK key, TV value, FineBstNode<TK, TV> parent)
        {
            Key = key;
            Value = value;
            Parent = parent;
        }
    }

    internal class FineBinaryTree<TK, TV> where TK : IComparable<TK>
    {
        internal FineBstNode<TK, TV> Root;
        private readonly Mutex _mutexRoot = new Mutex();
        internal FineBstNode<TK, TV> Find(TK key)
        {
            _mutexRoot?.WaitOne();
            var cur = Root;
            cur?.FineMutex.WaitOne();
            _mutexRoot?.ReleaseMutex();
            
            while (cur != null)
            {
                switch (cur.Key.CompareTo(key))
                {
                    //Eq
                    case 0:
                        cur.FineMutex.ReleaseMutex();
                        return cur;
                    //Greater
                    case 1:
                        cur.Left?.FineMutex.WaitOne();
                        cur.FineMutex.ReleaseMutex();
                        cur = cur.Left;
                        break;
                    // Less
                    case -1:
                        cur.Right?.FineMutex.WaitOne();
                        cur.FineMutex.ReleaseMutex();
                        cur = cur.Right;
                        break;
                }
            }
                
            return null;
        }

        internal void Insert(TK key, TV value)
        {
            _mutexRoot.WaitOne();
            var cur = Root;
            FineBstNode<TK, TV> parent = null;
            if (cur == null)
            {
                Root = new FineBstNode<TK, TV>(key, value);
                _mutexRoot.ReleaseMutex();
                return;
            }
            cur.FineMutex.WaitOne();
            _mutexRoot.ReleaseMutex();

            while (cur != null)
            {
                parent = cur;
                switch (cur.Key.CompareTo(key))
                {
                    //Eq
                    case 0:
                        cur.Value = value;
                        cur.FineMutex.ReleaseMutex();
                        return;
                    //Greater
                    case 1:
                        cur.Left?.FineMutex.WaitOne();
                        if(cur.Left != null) cur.FineMutex.ReleaseMutex();
                        cur = cur.Left;
                        break;
                    // Less
                    case -1:
                        cur.Right?.FineMutex.WaitOne();
                        if (cur.Right != null) cur.FineMutex.ReleaseMutex();
                        cur = cur.Right;
                        break;
                }
            }

            if (parent.Key.CompareTo(key) > 0)
            {
                parent.Left = new FineBstNode<TK, TV>(key, value, parent);
            }
            else
            {
                parent.Right = new FineBstNode<TK, TV>(key, value, parent);
            }
            parent.FineMutex.ReleaseMutex();
        }

        internal void Delete(TK key)
        {
            _mutexRoot.WaitOne();   
            var cur = Find(key);
            if (cur == null)
            {
                return;
            }
            cur.FineMutex.WaitOne();
            _mutexRoot.ReleaseMutex();
            
            var parent = cur.Parent;
            parent?.FineMutex.WaitOne();

            if (cur.Left == null && cur.Right == null)
            {
                if (parent == null)
                {
                    Root = null;
                    cur.FineMutex.ReleaseMutex();
                    return;
                }
                
                if (cur == parent.Left)
                {
                    parent.Left = null;
                }

                if (cur == parent.Right)
                {
                    parent.Right = null;
                }
                cur.FineMutex.ReleaseMutex();    
                parent.FineMutex.ReleaseMutex();
            }
            else if (cur.Left == null || cur.Right == null)
            {
                if (cur.Left == null)
                {
                    if (parent == null)
                    {
                        Root = cur.Right;
                        cur.Right.Parent = null;
                        cur.FineMutex.ReleaseMutex();
                        return;
                    }
                    if (cur == parent.Left)
                    {
                        parent.Left = cur.Right;
                    }
                    else
                    {
                        parent.Right = cur.Right;
                    }

                    cur.Right.Parent = parent;
                }
                else
                {
                    if (parent == null)
                    {
                        Root = cur.Left;
                        cur.Left.Parent = null;
                        cur.FineMutex.ReleaseMutex();
                        return;
                    }
                    
                    if (cur == parent.Left)
                    {
                        parent.Left = cur.Left;
                    }
                    else
                    {
                        parent.Right = cur.Left;
                    }

                    cur.Left.Parent = parent;
                }
                cur.FineMutex.ReleaseMutex();
                parent.FineMutex.ReleaseMutex();
            }
            else
            {
                var successor = Min(cur.Right);
                var sucParent = successor.Parent;
                cur.Key = successor.Key;
                cur.Value = successor.Value;
                if (successor == sucParent.Left)
                {
                    sucParent.Left = successor.Right;
                    if (successor.Right != null)
                    {
                        successor.Right.Parent = sucParent;
                    }
                }
                else
                {
                    sucParent.Right = successor.Right;
                    if (successor.Right != null)
                        successor.Right.Parent = sucParent;
                }
                cur.FineMutex.ReleaseMutex();
                parent?.FineMutex.ReleaseMutex();
            }
        }
        
        private static FineBstNode<TK, TV> Min(FineBstNode<TK, TV> cur)
        {
            while (true)
            {
                if (cur.Left == null) return cur;
                cur = cur.Left;
            }
        }
    }
}